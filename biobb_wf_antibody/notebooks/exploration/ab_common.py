"""Bound/unbound antibody case preparation, shared by the comparison notebooks.

Everything here is upstream of any actual measurement: fetching the entries, cleaning the
chains, IMGT-renumbering them with ANARCII, and establishing which residue of the bound
structure corresponds to which residue of the unbound one. `ab_rmsds.ipynb` measures
Cartesian RMSD on those pairs, `ab_torsions.ipynb` measures torsion differences; both need
exactly the same correspondence, and it must be the same one, not two implementations of it.

The functions were extracted verbatim from `ab_rmsds.ipynb` (sections 1-5), with two
additions:

* ANARCII results are cached as JSON next to the renumbered PDB, so a fresh kernel does not
  re-run 32 model inferences.
* `build_cases()` runs the whole preparation for the 16 cases and returns both the case
  contexts and the per-stage report tables, instead of the notebook doing it cell by cell.

`ab_rmsds.ipynb` still carries its own copies; it is the notebook that was written first and
its outputs on disk are what this module's PREP directory points at.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import warnings
from pathlib import Path

import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.Polypeptide import is_aa

# --- paths ----------------------------------------------------------------------------------
# PREP is shared with ab_rmsds.ipynb: it is that notebook's preparation output, and re-using it
# is what guarantees both notebooks measure the same structures. Results directories are not
# shared -- each notebook writes its own.
DATA = Path('data')
BASE = DATA / '0_base'
PREP = DATA / 'rmsds' / 'prep'

# --- IMGT definitions -----------------------------------------------------------------------
# https://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html
IMGT_CDRS = [(27, 38), (56, 65), (105, 117)]   # CDR1, CDR2, CDR3
IMGT_FV = (1, 128)                             # variable domain; beyond this ANARCII fabricates
IFACE_CUTOFFS = {'interface10': 10.0, 'interface5': 5.0}

AA3TO1 = {k.upper(): v for k, v in protein_letters_3to1.items()}

# --- the 16 cases ---------------------------------------------------------------------------
# Copied verbatim from biobb_antibody.ipynb.
#   column 1 = reference BOUND complex, '<code>_<antibody chains>:<antigen chains>'
#   column 2 = UNBOUND antibody
#   column 3 = UNBOUND antigen (unused here: the bound antigen comes from the reference entry)
# The chain letters are positional: RefAbAgs[i][0] chain j corresponds to RefAbAgs[i][1] chain j.
# They are NOT always H/L, and in 4GXU the chains named H and L are the *antigen*.
RefAbAgs = (
    # Reference         Antibody   Antigen
    ("2VXT_HL:I",	   "2VXU_HL", "1J0S_A"),
    ("2W9E_HL:A",	   "2W9D_HL", "1QM1_A"),
    ("3EOA_LH:I",	   "3EO9_LH", "3F74_A"),
    ("3HMX_LH:AB",	   "3HMW_LH", "1F45_AB"),
    ("3MXW_LH:A",	   "3MXV_LH", "3M1N_A"),
    ("5VPG_CD:A",	   "3RVT_CD", "3F5V_A"),
    ("4DN4_LH:M",	   "4DN3_LH", "1DOL_A"),
    ("4FQI_HL:ABEFCD", "4FQH_HL", "2FK0_AB"),
    ("4G6J_HL:A",      "4G5Z_HL", "4I1B_A"),
    ("4G6M_HL:A",      "4G6K_HL", "4I1B_A"),
    ("4GXU_MN:ABEFCD", "4GXV_HL", "1RUZ_HIJKLM"),
    # Medium
    ("3EO1_AB:CF",     "3EO0_AB", "1TGJ_AB"),
    ("3G6D_LH:A",      "3G6A_LH", "1IK0_A(10)"),
    ("3HI6_XY:B",      "3HI5_HL", "1MJN_A"),
    ("3L5W_LH:I",      "3L7E_LH", "1IK0_A(11)"),
    ("3V6Z_AB:F",      "3V6F_AB", "3KXS_F"),
)

# Published values, read out of https://zlab.wenglab.org/benchmark/Table_BM5.5.xlsx
# (columns 'I-RMSD (A)' and 'DASA(A2)'; 'difficulty' is the section the row sits in).
# Keyed by the UNBOUND antibody code, which is unique across the 16 cases -- the reference
# code is not usable as a key because RefAbAgs uses 5VPG where the benchmark row is 3RVW.
BENCHMARK = {
    #  unbound ab : (benchmark complex, difficulty, I-RMSD, dASA)
    '2VXU': ('2VXT_HL:I',       'rigid',  1.33, 2163),
    '2W9D': ('2W9E_HL:A',       'rigid',  1.13, 1677),
    '3EO9': ('3EOA_LH:I',       'rigid',  0.39, 1272),
    '3HMW': ('3HMX_LH:AB',      'rigid',  0.73, 1841),
    '3MXV': ('3MXW_LH:A',       'rigid',  0.48, 1696),
    '3RVT': ('3RVW_CD:A',       'rigid',  0.50, 1383),   # NOTE: RefAbAgs uses 5VPG, not 3RVW
    '4DN3': ('4DN4_LH:M',       'rigid',  0.81, 1317),
    '4FQH': ('4FQI_HL:ABEFCD',  'rigid',  1.08, 1459),
    '4G5Z': ('4G6J_HL:A',       'rigid',  0.61, 1893),
    '4G6K': ('4G6M_HL:A',       'rigid',  0.49, 1673),
    '4GXV': ('4GXU_MN:ABEFCD',  'rigid',  0.78, 1830),
    '3EO0': ('3EO1_AB:CF',      'medium', 1.37, 1630),
    '3G6A': ('3G6D_LH:A',       'medium', 1.86, 1793),
    '3HI5': ('3HI6_XY:B',       'medium', 1.65, 1871),
    '3L7E': ('3L5W_LH:I',       'medium', 0.48, 1138),
    '3V6F': ('3V6Z_AB:F',       'medium', 1.83, 1922),
}

CASE_NOTES = {
    '3RVT': 'reference 5VPG replaces the benchmark 3RVW (same complex, different crystal form)',
    '4FQH': 'antigen chains C,D,E,F are assembly-generated and absent from the 4FQI asymmetric unit',
}

_parser = PDBParser(QUIET=True)
warnings.filterwarnings('ignore', category=UserWarning, module='Bio.PDB')


# --- torsion definitions --------------------------------------------------------------------
# Shared by ab_torsions.ipynb (bound vs unbound crystal structures) and ab_ensemble.py (MD
# trajectories). They have to be the same tables in both places: the coverage metric of
# ab_ensemble compares an MD conformer with a crystal structure, and two different chi1 atom
# sets would make that comparison meaningless in a way nothing would flag.

CHI1_LAST = {          # 4th atom of chi1 = N, CA, CB, <this>
    'ARG': 'CG', 'ASN': 'CG', 'ASP': 'CG', 'CYS': 'SG', 'GLN': 'CG', 'GLU': 'CG',
    'HIS': 'CG', 'ILE': 'CG1', 'LEU': 'CG', 'LYS': 'CG', 'MET': 'CG', 'PHE': 'CG',
    'PRO': 'CG', 'SER': 'OG', 'THR': 'OG1', 'TRP': 'CG', 'TYR': 'CG', 'VAL': 'CG1',
}
CHI2_ATOMS = {         # full 4-atom set of chi2
    'ARG': ('CA', 'CB', 'CG', 'CD'), 'ASN': ('CA', 'CB', 'CG', 'OD1'),
    'ASP': ('CA', 'CB', 'CG', 'OD1'), 'GLN': ('CA', 'CB', 'CG', 'CD'),
    'GLU': ('CA', 'CB', 'CG', 'CD'), 'HIS': ('CA', 'CB', 'CG', 'ND1'),
    'ILE': ('CA', 'CB', 'CG1', 'CD1'), 'LEU': ('CA', 'CB', 'CG', 'CD1'),
    'LYS': ('CA', 'CB', 'CG', 'CD'), 'MET': ('CA', 'CB', 'CG', 'SD'),
    'PHE': ('CA', 'CB', 'CG', 'CD1'), 'PRO': ('CA', 'CB', 'CG', 'CD'),
    'TRP': ('CA', 'CB', 'CG', 'CD1'), 'TYR': ('CA', 'CB', 'CG', 'CD1'),
}
# chi angles whose terminal group has a true two-fold symmetry -> period 180, not 360
CHI_PERIOD = {('ASP', 'chi2'): 180.0, ('PHE', 'chi2'): 180.0, ('TYR', 'chi2'): 180.0}
# chi angles defined only up to which of two equivalent atoms got the '1' label
CHI_SWAP = {('VAL', 'chi1'): ('CG1', 'CG2'), ('LEU', 'chi2'): ('CD1', 'CD2')}

BB_ANGLES = ('phi', 'psi', 'omega')
CHI_ANGLES = ('chi1', 'chi2')
ALL_ANGLES = BB_ANGLES + CHI_ANGLES
PEPTIDE_BOND_MAX = 2.0        # A, C(i-1)-N(i); numbering cannot decide this in a crystal structure


def dihedral(p0, p1, p2, p3) -> float:
    """Torsion angle p0-p1-p2-p3 in degrees, IUPAC sign convention, in (-180, 180]."""
    import numpy as np

    b0 = np.asarray(p0, float) - np.asarray(p1, float)
    b1 = np.asarray(p2, float) - np.asarray(p1, float)
    b2 = np.asarray(p3, float) - np.asarray(p2, float)
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1          # b0 and b2 projected onto the plane normal to b1
    w = b2 - np.dot(b2, b1) * b1
    from math import degrees
    return degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def wrap(d, period: float = 360.0):
    """Map an angle difference into (-period/2, period/2]."""
    import numpy as np

    d = np.asarray(d, float)
    return -(-(d + period / 2) % period - period / 2)


def chi_quads(resname: str, alt: bool = False) -> dict:
    """{'chi1': (4 atom names), 'chi2': ...} for a residue type, or the swapped naming."""
    name = resname.upper()
    out = {}
    if name in CHI1_LAST:
        out['chi1'] = ('N', 'CA', 'CB', CHI1_LAST[name])
    if name in CHI2_ATOMS:
        out['chi2'] = CHI2_ATOMS[name]
    if alt:
        for angle, (a, b) in CHI_SWAP.items():
            if angle[0] == name and angle[1] in out:
                out[angle[1]] = tuple(b if x == a else x for x in out[angle[1]])
    return out


# --- 1. identifiers and entries -------------------------------------------------------------

def parse_identifier(identifier: str):
    """Split an identifier of the RefAbAgs list into its parts.

    The chains of interest follow the PDB code, the ones after the colon are the
    antigen chains of a reference complex. A trailing '(<n>)' is the model to extract
    from the entry, needed by the NMR ensembles that hold one conformer per model:
      '4G6K_HL'    -> ('4G6K', ['H', 'L'], [],    None)
      '4G6M_HL:A'  -> ('4G6M', ['H', 'L'], ['A'], None)
      '1IK0_A(10)' -> ('1IK0', ['A'],      [],    '10')
    """
    model = re.search(r'\((\d+)\)$', identifier.strip())
    if model:
        identifier = identifier[:model.start()]
    pdb_code, _, chains = identifier.partition('_')
    before_colon, _, after_colon = chains.partition(':')
    return pdb_code, list(before_colon), list(after_colon), model.group(1) if model else None


def entry_pdb_path(identifier: str) -> Path:
    return BASE / f'{parse_identifier(identifier)[0]}.pdb'


def fetch_entries(verbose: bool = True) -> list[str]:
    """Download any of the 32 required PDB entries that is not already in BASE."""
    from biobb_io.api.pdb import pdb

    downloaded = []
    for ref, ab, _ag in RefAbAgs:
        for ident in (ref, ab):
            path = entry_pdb_path(ident)
            if not path.is_file():
                code = parse_identifier(ident)[0]
                if verbose:
                    print(f'downloading {code} ...')
                path.parent.mkdir(parents=True, exist_ok=True)
                pdb(output_pdb_path=str(path), properties={'pdb_code': code, 'restart': True})
                downloaded.append(code)
    for ref, ab, _ag in RefAbAgs:
        assert entry_pdb_path(ref).is_file() and entry_pdb_path(ab).is_file(), (ref, ab)
    return downloaded


# --- 2. structure preparation ---------------------------------------------------------------

def pdb_tools_pipeline(inp_file, out_file, steps):
    """Concatenate pdb_tools calls, chaining each step's output into the next.

    The intermediate file is unique and lives next to out_file instead of being a fixed
    'tmp.pdb' in the working directory, so concurrent/repeated calls cannot clobber each other.
    """
    out_file = str(out_file)
    tmp_file = f'{out_file}.pipe.tmp.pdb'
    current = str(inp_file)
    try:
        for step, props in steps:
            step(input_file_path=current, output_file_path=out_file, properties=props)
            os.replace(out_file, tmp_file)
            current = tmp_file
        os.replace(current, out_file)
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


def chains_present(pdb_path) -> list[str]:
    """Chain identifiers that actually carry ATOM records, in order of first appearance."""
    seen = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(('ATOM  ', 'HETATM')):
                ch = line[21]
                if ch not in seen:
                    seen.append(ch)
    return seen


def prepare_chains(src, out, chains: list[str], model: str | None = None):
    """Extract `chains` from `src` into `out`, keeping chain IDs, numbering and insertion codes."""
    from biobb_pdb_tools.pdb_tools import (biobb_pdb_delhetatm, biobb_pdb_keepcoord,
                                          biobb_pdb_selaltloc, biobb_pdb_selchain,
                                          biobb_pdb_selmodel, biobb_pdb_tidy)

    steps = [(biobb_pdb_selmodel.biobb_pdb_selmodel, {'models': model})] if model else []
    steps += [
        (biobb_pdb_tidy.biobb_pdb_tidy,           {'strict': True}),                 # 1. format-clean
        (biobb_pdb_selchain.biobb_pdb_selchain,   {'chains': ','.join(chains)}),     # 2. keep chains
        (biobb_pdb_delhetatm.biobb_pdb_delhetatm, {}),                               # 3. drop HETATM/waters
        (biobb_pdb_selaltloc.biobb_pdb_selaltloc, {}),                               # 4. highest-occupancy altloc
        (biobb_pdb_keepcoord.biobb_pdb_keepcoord, {}),                               # 5. coordinates only
        (biobb_pdb_tidy.biobb_pdb_tidy,           {'strict': True}),                 # 6. re-emit TER/END
    ]
    pdb_tools_pipeline(src, out, steps)


def case_paths(idx: int, ref: str) -> dict:
    d = PREP / f'{idx:02d}_{parse_identifier(ref)[0]}'
    return {'dir': d, 'ref_ab': d / 'ref_ab.pdb', 'ref_ag': d / 'ref_ag.pdb',
            'unb_ab': d / 'unb_ab.pdb'}


def prepare_case(idx: int, ref: str, unb: str, force: bool = False) -> dict:
    """Produce the three cleaned PDBs for one case and report what was done."""
    ref_code, ref_ab_chains, ref_ag_chains, ref_model = parse_identifier(ref)
    unb_code, unb_ab_chains, _, unb_model = parse_identifier(unb)
    paths = case_paths(idx, ref)
    paths['dir'].mkdir(parents=True, exist_ok=True)

    warns = []
    if len(ref_ab_chains) != len(unb_ab_chains):
        raise ValueError(f'antibody chain count differs: {ref_ab_chains} vs {unb_ab_chains}')

    ref_src, unb_src = entry_pdb_path(ref), entry_pdb_path(unb)
    avail_ref, avail_unb = chains_present(ref_src), chains_present(unb_src)
    for label, want, have in (('ref antibody', ref_ab_chains, avail_ref),
                              ('unb antibody', unb_ab_chains, avail_unb)):
        if set(want) - set(have):
            raise ValueError(f'{label} chains {sorted(set(want) - set(have))} absent from the entry')

    # Requested antigen chains are not always in the asymmetric unit (4FQI: C,D,E,F come from
    # the biological assembly). Use what is there and record the rest.
    ag_found = [c for c in ref_ag_chains if c in avail_ref]
    ag_missing = [c for c in ref_ag_chains if c not in avail_ref]
    if ag_missing:
        warns.append(f'antigen chains {",".join(ag_missing)} absent from {ref_code}')
    if not ag_found:
        raise ValueError(f'no antigen chain of {ref_ag_chains} present in {ref_code}')

    todo = [(paths['ref_ab'], ref_src, ref_ab_chains, ref_model),
            (paths['ref_ag'], ref_src, ag_found,      ref_model),
            (paths['unb_ab'], unb_src, unb_ab_chains, unb_model)]
    for out, src, chains, model in todo:
        if force or not out.is_file():
            prepare_chains(src, out, chains, model)

    return {'case_idx': idx, 'ref': ref, 'unb': unb, 'ref_code': ref_code, 'unb_code': unb_code,
            'ref_ab_chains': ref_ab_chains, 'unb_ab_chains': unb_ab_chains,
            'ag_chains': ag_found, 'missing_ag_chains': ','.join(ag_missing),
            'iface_partial': bool(ag_missing), 'paths': paths, 'warnings': warns}


# --- 3. observed residues and IMGT numbering ------------------------------------------------

_obs_cache: dict[str, dict] = {}


def observed_residues(pdb_path) -> dict:
    """{chain_id: (one_letter_seq, [Residue, ...])} for amino acids, in file order.

    Memoised, so every stage (IMGT map, alignment, interface, measurement) works on the *same*
    Residue objects for a given file and residues can be compared by identity.
    """
    key = str(pdb_path)
    if key not in _obs_cache:
        structure = _parser.get_structure('s', key)
        out = {}
        for chain in structure[0]:
            res = [r for r in chain if is_aa(r, standard=False)]
            if res:
                seq = ''.join(AA3TO1.get(r.get_resname().upper(), 'X') for r in res)
                out[chain.id] = (seq, res)
        _obs_cache[key] = out
    return _obs_cache[key]


def res_key(residue) -> tuple:
    """(chain_id, hetflag, resseq, icode) -- stable identity of a residue within its file."""
    return (residue.get_parent().id, *residue.id)


def imgt_region(num: int) -> str:
    """IMGT position -> 'CDR1' | 'CDR2' | 'CDR3' | 'FR' | 'const'."""
    if num > IMGT_FV[1]:
        return 'const'
    for i, (start, end) in enumerate(IMGT_CDRS, start=1):
        if start <= num <= end:
            return f'CDR{i}'
    return 'FR'


def _anarcii_to_json(results: dict) -> list:
    return [{'model': model, 'chain': chain, 'chain_type': e['chain_type'],
             'score': float(e['score']), 'query_start': int(e['query_start']),
             'numbering': [[int(n), ic, aa] for (n, ic), aa in e['numbering']]}
            for (model, chain), e in results.items()]


def _anarcii_from_json(blob: list) -> dict:
    return {(e['model'], e['chain']):
            {'chain_type': e['chain_type'], 'score': e['score'], 'query_start': e['query_start'],
             'numbering': [((n, ic), aa) for n, ic, aa in e['numbering']]}
            for e in blob}


def run_anarcii(pdb_in, pdb_out, force: bool = False) -> dict:
    """Renumber `pdb_in` to IMGT and return ANARCII's {(model, chain_id): entry} results.

    The results are cached as JSON beside `pdb_out`; ANARCII is a neural model and re-running
    it 32 times on every kernel restart is a minute of nothing.
    """
    from biobb_haddock.utils.anarcii import Anarcii

    cache = Path(str(pdb_out).replace('.pdb', '') + '_anarcii.json')
    if not force and cache.is_file() and Path(pdb_out).is_file():
        return _anarcii_from_json(json.loads(cache.read_text()))

    bb = Anarcii(input_pdb_path=str(pdb_in), output_pdb_path=str(pdb_out),
                 properties={'seq_type': 'antibody', 'mode': 'accuracy', 'verbose': False,
                             'restart': False})
    bb.launch()
    cache.write_text(json.dumps(_anarcii_to_json(bb.results)))
    return bb.results


def imgt_map(pdb_path, results: dict) -> dict:
    """Zip ANARCII's numbering onto the observed residues of `pdb_path`.

    Returns {chain_id: {'chain_type', 'score', 'ok', 'map': {(imgt_num, icode): Residue}}}.
    Only IMGT positions 1-128 are kept: beyond the numbered domain ANARCII fabricates
    sequential numbers, which are not comparable between two structures.
    """
    observed = observed_residues(pdb_path)
    out = {}
    for (_model, chain_id), entry in results.items():
        seq, residues = observed[chain_id]
        numbered = [(num, aa) for num, aa in entry['numbering'] if aa != '-']
        numbered_seq = ''.join(aa for _num, aa in numbered)
        offset = seq.find(numbered_seq)
        if offset < 0:                      # same fallback ANARCII's renumber_pdbx uses
            offset = entry['query_start']
        mapping = {}
        for i, ((num, icode), aa) in enumerate(numbered):
            residue = residues[offset + i]
            got = AA3TO1.get(residue.get_resname().upper(), 'X')
            if got != aa:
                raise ValueError(f'{pdb_path} chain {chain_id}: ANARCII residue {num}{icode} is '
                                 f'{aa} but the structure has {got}')
            if IMGT_FV[0] <= num <= IMGT_FV[1]:
                mapping[(num, icode.strip())] = residue
        # ANARCII's own QA gate: a domain it could not confidently number is left un-renumbered
        ok = entry['chain_type'] in 'HLKABDG' and entry['score'] >= 19 and (104, '') in mapping
        out[chain_id] = {'chain_type': entry['chain_type'], 'score': float(entry['score']),
                         'ok': bool(ok), 'map': mapping,
                         'n_obs': len(seq), 'n_numbered': len(numbered)}
    return out


# --- 4. residue correspondence --------------------------------------------------------------

_aligner = PairwiseAligner(mode='global',
                           substitution_matrix=substitution_matrices.load('BLOSUM62'),
                           open_gap_score=-11, extend_gap_score=-1)   # blastp-like


def align_chains(seq_a: str, seq_b: str):
    """Global-align two sequences; return (index pairs, identity, coverage, score)."""
    aln = _aligner.align(seq_a.replace('X', 'A'), seq_b.replace('X', 'A'))[0]
    pairs, identical = [], 0
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        for k in range(a1 - a0):
            ia, ib = a0 + k, b0 + k
            pairs.append((ia, ib))
            identical += seq_a[ia] == seq_b[ib]
    shortest = min(len(seq_a), len(seq_b))
    return pairs, identical / shortest, len(pairs) / shortest, float(aln.score)


def build_correspondence(ctx: dict) -> dict:
    """Match bound and unbound antibody residues chain by chain, by sequence alignment.

    Returns {'pairs': [(slot, ref_residue, unb_residue), ...], 'chains': [per-chain stats]}.
    """
    ctx['warn_corr'] = []          # reset, so re-running is idempotent
    ref_obs = observed_residues(ctx['paths']['ref_ab'])
    unb_obs = observed_residues(ctx['paths']['unb_ab'])
    ref_chains, unb_chains = ctx['ref_ab_chains'], ctx['unb_ab_chains']

    # Re-derive the pairing from the sequences alone and check it against the positional one
    if len(unb_chains) <= 3:
        best = max(itertools.permutations(range(len(unb_chains))),
                   key=lambda perm: sum(align_chains(ref_obs[ref_chains[i]][0],
                                                     unb_obs[unb_chains[p]][0])[3]
                                        for i, p in enumerate(perm)))
        if list(best) != list(range(len(unb_chains))):
            ctx['warn_corr'].append(
                f'sequence-derived chain pairing {best} differs from the positional one; using it')
            unb_chains = [unb_chains[p] for p in best]
            ctx['unb_ab_chains_used'] = unb_chains

    pairs, chain_stats = [], []
    for slot, (rc, uc) in enumerate(zip(ref_chains, unb_chains)):
        rseq, rres = ref_obs[rc]
        useq, ures = unb_obs[uc]
        idx_pairs, identity, coverage, _score = align_chains(rseq, useq)
        if identity < 0.80:
            raise ValueError(f'slot {slot} ({rc} vs {uc}): identity {identity:.2f} is too low to '
                             f'be the same antibody chain')
        if identity < 0.98:
            ctx['warn_corr'].append(f'slot {slot} ({rc} vs {uc}): sequence identity {identity:.3f}')
        pairs += [(slot, rres[ia], ures[ib]) for ia, ib in idx_pairs]
        chain_stats.append({'slot': slot, 'chain_ref': rc, 'chain_unb': uc,
                            'chain_type': ctx['ref_imgt'][rc]['chain_type'],
                            'len_ref': len(rseq), 'len_unb': len(useq),
                            'identity': identity, 'coverage': coverage,
                            'n_mutations': sum(rseq[ia] != useq[ib] for ia, ib in idx_pairs)})
    return {'pairs': pairs, 'chains': chain_stats}


def imgt_pairs(ctx: dict) -> list:
    """Matched Fv residue pairs from the IMGT maps: [(slot, imgt_key, ref_residue, unb_residue)]."""
    out = []
    unb_chains = ctx.get('unb_ab_chains_used', ctx['unb_ab_chains'])
    for slot, (rc, uc) in enumerate(zip(ctx['ref_ab_chains'], unb_chains)):
        rinfo, uinfo = ctx['ref_imgt'][rc], ctx['unb_imgt'][uc]
        if not (rinfo['ok'] and uinfo['ok']):
            continue
        for key in sorted(rinfo['map'].keys() & uinfo['map'].keys()):
            out.append((slot, key, rinfo['map'][key], uinfo['map'][key]))
    return out


def merged_pairs(ctx: dict) -> list:
    """The `full` correspondence: IMGT inside the Fv, alignment everywhere else.

    Where a one-residue indel falls inside a run of identical residues (3V6Z/3V6F, CDR2), the
    alignment's gap placement is score-degenerate and therefore arbitrary, while IMGT places it
    by structural convention. Letting IMGT win inside the Fv keeps the `full` extent consistent
    with the `fv`/CDR extents instead of pairing a handful of residues two different ways.
    """
    pairs = [(slot, r, u) for slot, _key, r, u in ctx['imgt_pairs']]
    used_ref = {res_key(r) for _s, r, _u in pairs}
    used_unb = {res_key(u) for _s, _r, u in pairs}
    for slot, r, u in ctx['corr']['pairs']:
        if res_key(r) not in used_ref and res_key(u) not in used_unb:
            pairs.append((slot, r, u))
    return pairs


# --- 5. interface residues ------------------------------------------------------------------

def heavy_atoms(residues) -> list:
    return [a for r in residues for a in r if a.element != 'H']


def interface_residues(ab_residues, ag_residues, cutoff: float) -> set:
    """Antibody residues with any heavy atom within `cutoff` of an antigen heavy atom."""
    ag_atoms = heavy_atoms(ag_residues)
    if not ag_atoms:
        return set()
    search = NeighborSearch(ag_atoms)
    hits = set()
    for residue in ab_residues:
        for atom in residue:
            if atom.element == 'H':
                continue
            if search.search(atom.coord, cutoff, level='A'):
                hits.add(res_key(residue))
                break
    return hits


# --- orchestration --------------------------------------------------------------------------

def build_cases(force_prep: bool = False, force_anarcii: bool = False,
                verbose: bool = True) -> tuple[dict, dict]:
    """Run sections 1-5 for all 16 cases.

    Returns (cases, tables) where `cases` maps case index to its context dict and `tables`
    holds the per-stage report DataFrames ('prep', 'anarcii', 'align', 'iface', 'case').
    """
    fetch_entries(verbose=verbose)

    cases, prep_rows = {}, []
    for idx, (ref, ab, _ag) in enumerate(RefAbAgs):
        try:
            ctx = prepare_case(idx, ref, ab, force=force_prep)
            cases[idx] = ctx
            status = 'ok'
        except Exception as exc:
            ctx = {'case_idx': idx, 'ref': ref, 'unb': ab, 'warnings': []}
            status = f'{type(exc).__name__}: {exc}'
        prep_rows.append({
            'case': idx, 'ref': ref, 'unb': ab,
            'ab_chains': (''.join(ctx.get('ref_ab_chains', [])) + '/'
                          + ''.join(ctx.get('unb_ab_chains', []))),
            'ag_chains': ''.join(ctx.get('ag_chains', [])),
            'missing_ag': ctx.get('missing_ag_chains', ''),
            'status': status,
        })

    # --- IMGT numbering
    anarcii_rows = []
    for idx, ctx in cases.items():
        ctx['warn_anarcii'] = []
        paths = ctx['paths']
        for which, chains in (('ref', ctx['ref_ab_chains']), ('unb', ctx['unb_ab_chains'])):
            pdb_imgt = paths['dir'] / f'{which}_ab_imgt.pdb'
            ctx[f'{which}_anarcii'] = run_anarcii(paths[f'{which}_ab'], pdb_imgt,
                                                  force=force_anarcii)
            paths[f'{which}_ab_imgt'] = pdb_imgt
            ctx[f'{which}_imgt'] = imgt_map(paths[f'{which}_ab'], ctx[f'{which}_anarcii'])
            for slot, chain_id in enumerate(chains):
                info = ctx[f'{which}_imgt'][chain_id]
                anarcii_rows.append({'case': idx, 'which': which, 'slot': slot, 'chain': chain_id,
                                     'type': info['chain_type'], 'score': round(info['score'], 1),
                                     'ok': info['ok'], 'n_obs': info['n_obs'],
                                     'n_fv': len(info['map'])})
        # The slot pairing must agree with ANARCII's own H/L call, an independent signal
        for slot, (rc, uc) in enumerate(zip(ctx['ref_ab_chains'], ctx['unb_ab_chains'])):
            rt, ut = ctx['ref_imgt'][rc]['chain_type'], ctx['unb_imgt'][uc]['chain_type']
            if (rt == 'H') != (ut == 'H'):
                ctx['warn_anarcii'].append(
                    f'slot {slot}: chain_type {rt} (bound {rc}) vs {ut} (unbound {uc})')

    # --- correspondence
    align_rows = []
    for idx, ctx in cases.items():
        ctx['corr'] = build_correspondence(ctx)
        ctx['align_pairs'] = {(res_key(r), res_key(u)) for _s, r, u in ctx['corr']['pairs']}
        ctx['imgt_pairs'] = imgt_pairs(ctx)
        ctx['imgt_vs_align_disagree'] = sum(
            1 for _slot, _key, r, u in ctx['imgt_pairs']
            if (res_key(r), res_key(u)) not in ctx['align_pairs'])
        ctx['full_pairs'] = merged_pairs(ctx)
        # heavy vs light comes from ANARCII's chain_type, not from the chain letter
        ctx['slot_is_heavy'] = {slot: ctx['ref_imgt'][ch]['chain_type'] == 'H'
                                for slot, ch in enumerate(ctx['ref_ab_chains'])}
        align_rows += [{'case': idx, **cs} for cs in ctx['corr']['chains']]

    # --- interface
    iface_rows = []
    for idx, ctx in cases.items():
        ag_res = [r for _seq, res in observed_residues(ctx['paths']['ref_ag']).values() for r in res]
        all_ab = [r for _seq, res in observed_residues(ctx['paths']['ref_ab']).values() for r in res]
        matched = {res_key(r) for _slot, r, _u in ctx['full_pairs']}
        ctx['n_ag_res'] = len(ag_res)
        row = {'case': idx, 'ref': ctx['ref'], 'n_ag_res': len(ag_res),
               'partial_ag': ctx['iface_partial']}
        for name, cutoff in IFACE_CUTOFFS.items():
            detected = interface_residues(all_ab, ag_res, cutoff)
            keys = detected & matched
            ctx[name] = keys
            ctx[f'{name}_unmapped'] = len(detected) - len(keys)
            row[f'{name}_n'] = len(keys)
            row[f'{name}_unmapped'] = ctx[f'{name}_unmapped']
            if detected and ctx[f'{name}_unmapped'] / len(detected) > 0.10:
                ctx['warn_corr'].append(f'{name}: {ctx[f"{name}_unmapped"]}/{len(detected)} '
                                        f'interface residues have no unbound counterpart')
        assert ctx['interface5'] <= ctx['interface10'], f'case {idx}: 5 A iface not inside 10 A'
        iface_rows.append(row)

    # --- one row per case, carrying every provenance flag downstream analyses need
    case_rows = []
    for idx, ctx in cases.items():
        bench = BENCHMARK[ctx['unb_code']]
        case_rows.append({
            'case_idx': idx, 'ref': ctx['ref'], 'unb': ctx['unb'],
            'ref_code': ctx['ref_code'], 'unb_code': ctx['unb_code'],
            'benchmark_complex': bench[0], 'difficulty': bench[1],
            'benchmark_irmsd': bench[2], 'benchmark_dasa': bench[3],
            'benchmark_ref_mismatch': bench[0].split('_')[0] != ctx['ref_code'],
            'chain_map': ','.join(f'{r}->{u}' for r, u in
                                  zip(ctx['ref_ab_chains'],
                                      ctx.get('unb_ab_chains_used', ctx['unb_ab_chains']))),
            'chain_types': ','.join(ctx['ref_imgt'][c]['chain_type'] for c in ctx['ref_ab_chains']),
            'anarcii_min_score': min(ctx['ref_imgt'][c]['score'] for c in ctx['ref_ab_chains']),
            'anarcii_all_ok': (all(ctx['ref_imgt'][c]['ok'] for c in ctx['ref_ab_chains'])
                               and all(ctx['unb_imgt'][c]['ok'] for c in ctx['unb_ab_chains'])),
            'min_identity': min(c['identity'] for c in ctx['corr']['chains']),
            'min_coverage': min(c['coverage'] for c in ctx['corr']['chains']),
            'n_mutations': sum(c['n_mutations'] for c in ctx['corr']['chains']),
            'n_fv': len(ctx['imgt_pairs']), 'n_full': len(ctx['full_pairs']),
            'n_iface10': len(ctx['interface10']), 'n_iface5': len(ctx['interface5']),
            'imgt_vs_align_disagree': ctx['imgt_vs_align_disagree'],
            'missing_ag_chains': ctx['missing_ag_chains'], 'iface_partial': ctx['iface_partial'],
            'note': CASE_NOTES.get(ctx['unb_code'], ''),
            'warnings': ' | '.join(ctx['warnings'] + ctx['warn_anarcii'] + ctx['warn_corr']),
        })

    tables = {'prep': pd.DataFrame(prep_rows).set_index('case'),
              'anarcii': pd.DataFrame(anarcii_rows),
              'align': pd.DataFrame(align_rows),
              'iface': pd.DataFrame(iface_rows).set_index('case'),
              'case': pd.DataFrame(case_rows)}
    return cases, tables
