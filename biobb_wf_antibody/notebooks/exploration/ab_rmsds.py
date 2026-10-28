"""Helpers of ab_rmsds.ipynb: how far the unbound antibody of each benchmark case is from its bound form.

The structures are the workflow's own. Every case is prepared by stage 1 from the configuration
array/launch_wf.py writes for it, so what is measured is what HADDOCK3 docks: the assembly or the
model stage 0 downloads, the unbound antibody and antigen repaired by Modeller, and the reference
complex fused into chain A (antibody) and chain B (antigen). The residue correspondence is
utils.map_contact_residues, the interface is haddock_interface and the CDRs come from ANARCII
through cdr.imgt_region_ri, the calls stages 1 and 2 make.

What is left here is what only the notebook needs: the published benchmark values, the I-RMSD as
the benchmark defines it, the RMSD of the antibody regions, the per-case tables, the checks, the
docking results read back next to them (with the cluster structures the CDR ensembles docked), and
the figures. The notebook calls these and shows what they return.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tarfile
import urllib.request
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.SeqUtils import seq1
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import stats
from scipy.spatial import cKDTree
from biobb_common import biobb_global_properties
from biobb_common.configuration import settings
from biobb_common.tools import file_utils as fu
from biobb_haddock.haddock_restraints.haddock_interface import haddock_interface
from biobb_haddock.utils.anarcii import Anarcii

from antibody_wf import cdr, utils
from antibody_wf.stages.haddock import (merge_structures, prepare_antibody, prepare_antigen,
                                        prepare_structures)
from antibody_wf.workflow import download_pdbs

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'biobb_wf_antibody' / 'array'))
from launch_wf import COMPLEXES, write_case_config  # noqa: E402

# The building blocks call pdb_tools, check_structure and haddock-restraints as executables, and a
# kernel started without activating the environment does not have them on its PATH
_ENV_BIN = str(Path(sys.executable).parent)
if _ENV_BIN not in os.environ.get('PATH', '').split(os.pathsep):
    os.environ['PATH'] = _ENV_BIN + os.pathsep + os.environ.get('PATH', '')
# Every building block of the preparation logs to its step folder only, not to the notebook
biobb_global_properties.update({'can_write_console_log': False})

OUT = ROOT / 'output' / 'exploration' / 'rmsds'
PREP = OUT / 'prep'
RES = OUT / 'results'
FIG = OUT / 'figures'
# Shared with tests/test_benchmark_structures.py, which fetches the same files
BENCHMARK_DIR = ROOT / 'output' / 'benchmark5.5'
BENCHMARK_URL = 'https://zlab.wenglab.org/benchmark/benchmark5.5.tgz'

# "The I-RMSD was calculated by superposing the unbound component proteins onto their bound
# forms, using the Ca atoms for residues that had any atom within 10 A of any atom of the
# binding partner." (Docking Benchmark 4/5)
IFACE_CUTOFF = 10.0

# Published values, read out of https://zlab.wenglab.org/benchmark/Table_BM5.5.xlsx
# (columns 'I-RMSD (A)' and 'dASA(A2)'; 'difficulty' is the section the row sits in). Keyed by
# the unbound antibody code, unique across the 16 cases: the reference code is not usable as a
# key because the workflow uses 5VPG where the benchmark row is 3RVW.
BENCHMARK = {
    #  unbound ab : (benchmark complex, difficulty, I-RMSD, dASA)
    '2VXU': ('2VXT_HL:I',       'rigid',  1.33, 2163),
    '2W9D': ('2W9E_HL:A',       'rigid',  1.13, 1677),
    '3EO9': ('3EOA_LH:I',       'rigid',  0.39, 1272),
    '3HMW': ('3HMX_LH:AB',      'rigid',  0.73, 1841),
    '3MXV': ('3MXW_LH:A',       'rigid',  0.48, 1696),
    '3RVT': ('3RVW_CD:A',       'rigid',  0.50, 1383),
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
}

# Region sets of the antibody, as residue indices of the prepared chain A. The CDR and framework
# ranges are cdr.py's, the ones stage 2 builds its index groups from.
CDRS = ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']
EXTENTS = ['fv', 'framework', 'cdr_all', *CDRS, 'full', 'interface']
# scheme -> extent whose Ca atoms are fitted, None fits the measured atoms themselves
SCHEMES = {'self': None, 'fv_fw': 'framework', 'full': 'full'}
# A least-squares fit on a handful of atoms is ill-defined; measuring them in a frame fitted on
# other atoms is not. IMGT CDR-L2 is 3 residues wide in all 16 antibodies.
MIN_FIT_RES = 4
# The V domain ends at IMGT 128; cdr.FRAMEWORK_RANGES runs to 129, which ANARCII gives to the
# first CH1/CL residue. That residue follows the elbow (7.9 A off in 3EOA once the framework is
# fitted) and took the framework self RMSD of 3EOA from 0.67 to 0.89 A, so it is left out here.
IMGT_LAST = 128


def benchmark_row(idx):
    """(benchmark complex, difficulty, I-RMSD, dASA) of case `idx`."""
    return BENCHMARK[utils.parse_identifier(COMPLEXES[idx][1])[0]]


def benchmark_code(idx):
    return benchmark_row(idx)[0].split('_')[0]


def case_info(idx):
    """Identifiers of case `idx` and its row of the benchmark table."""
    reference, antibody, antigen = COMPLEXES[idx]
    complex_, difficulty, published, dasa = benchmark_row(idx)
    unb_code = utils.parse_identifier(antibody)[0]
    return {'case_idx': idx, 'reference': reference, 'antibody': antibody, 'antigen': antigen,
            'ref_code': utils.parse_identifier(reference)[0], 'unb_code': unb_code,
            'benchmark_complex': complex_, 'difficulty': difficulty,
            'benchmark_irmsd': published, 'benchmark_dasa': dasa,
            'note': CASE_NOTES.get(unb_code, '')}


# ============================================================================
# Preparation
# ============================================================================

def _modeller_key():
    """MODELLER_KEY, or the key file tests/test_repair_structure.py reads."""
    key_file = ROOT / 'biobb_wf_antibody' / 'config' / 'modeller_key.txt'
    return os.environ.get('MODELLER_KEY') or (key_file.read_text().strip() if key_file.is_file() else None)


_CONFS = {}


def case_config(idx):
    """Logger, properties, paths and complex of case `idx`, read as workflow.py reads them.

    The configuration is the one array/launch_wf.py writes for a cluster run, so every path and
    property below is the run's own. It is rewritten once per session, so an identifier changed
    in COMPLEXES (a pinned model, say) reaches it.
    """
    if idx not in _CONFS:
        case_dir = PREP / f'case_{idx}'
        config = case_dir / 'workflow.yml'
        with contextlib.redirect_stdout(io.StringIO()):
            write_case_config(idx, str(PREP))
        conf = settings.ConfReader(str(config))
        global_log, _ = fu.get_logs(path=str(case_dir), light_format=True, can_write_console=False)
        global_prop = conf.get_prop_dic(global_log=global_log)
        _CONFS[idx] = {'dir': case_dir, 'log': global_log, 'prop': global_prop,
                       'paths': conf.get_paths_dic(),
                       'ids': utils.resolve_complex(global_prop['step0_0_pdb_codes'])}
    return _CONFS[idx]


def case_files(idx):
    """The files of case `idx` the analysis reads, named after what they hold."""
    c = case_config(idx)
    paths = c['paths']
    return {
        'reference_entry': paths['step0_1_pdb_reference']['output_pdb_path'],
        'antibody_entry': paths['step0_2_pdb_antibody']['output_pdb_path'],
        'antigen_entry': paths['step0_3_pdb_antigen']['output_pdb_path'],
        'antibody_fixed': paths['step1_repair_antibody']['output_pdb_path'],
        'antigen_fixed': paths['step1_repair_antigen']['output_pdb_path'],
        'antibody': paths['step1_0_prepare_antibody']['output_pdb_path'],      # chain A
        # the same residues, in the same order, before they are fused into chain A
        'antibody_merged': paths['step1_1_biobb_pdb_merge_antibody']['output_file_path'],
        'reference_antibody_merged': paths['step1_4_biobb_pdb_merge_reference_antibody']['output_file_path'],
        'antigen': paths['step1_2_prepare_antigen']['output_pdb_path'],        # chain B
        'reference': paths['step1_6_merge_reference_complex']['output_pdb_path'],  # A + B
        'anarcii': paths['step2_19_anarcii']['output_pdb_path'],
        'interface': str(c['dir'] / 'analysis' / f'interface_{IFACE_CUTOFF:g}A.txt'),
    }


def prepare_case(idx, force=False):
    """Download and prepare case `idx` as stages 0 and 1 do, then number its antibody (step2_19).

    Done once per set of identifiers: the outputs are kept until `force` or until the case's
    entry in COMPLEXES differs from the one they were prepared from (prepared_from.json).
    Returns case_files(idx).
    """
    c = case_config(idx)
    files = case_files(idx)
    stamp = c['dir'] / 'prepared_from.json'
    prepared = ('antibody', 'antigen', 'reference', 'interface')
    stale = (force or not stamp.is_file() or json.loads(stamp.read_text()) != list(COMPLEXES[idx])
             or not all(Path(files[k]).is_file() for k in prepared))
    if _modeller_key():
        os.environ.setdefault('MODELLER_KEY', _modeller_key())
    # biobb writes its log files and sandboxes to the working directory
    with fu.change_dir(str(c['dir'])):
        if stale:
            # the configuration enables restart, which would keep the outputs of the old entry
            prop = {step: dict(p, restart=False) if isinstance(p, dict) else p for step, p in c['prop'].items()}
            download_pdbs(c['log'], prop, c['paths'], c['ids'])
            prepare_structures(c['log'], prop, c['paths'], c['ids'])
            Path(files['interface']).unlink(missing_ok=True)
            interface_residues(files['reference'], files['interface'])
        anarcii_chain_types(c['paths']['step2_19_anarcii'], c['prop']['step2_19_anarcii'], stale)
    stamp.write_text(json.dumps(list(COMPLEXES[idx])))
    return files


def prepare_benchmark_case(idx, force=False):
    """The benchmark's own bound and unbound files of case `idx`, through steps 1_0 to 1_6.

    No download and no repair: only the cleaning and the fusion into chains A and B, so that the
    I-RMSD code reads them exactly as it reads the workflow's structures. The chains are taken in
    file order, which is how the benchmark pairs them ('r' is the antibody, 'l' the antigen).
    """
    code = benchmark_code(idx)
    fetch_benchmark([code])
    src = {f'{side}_{state}': BENCHMARK_DIR / f'{code}_{side}_{state}.pdb' for side in 'rl' for state in 'bu'}
    d = PREP / f'case_{idx}' / 'benchmark'
    files = {'antibody': d / 'r_u' / 'antibody_clean.pdb', 'antigen': d / 'l_u' / 'antigen_clean.pdb',
             'reference': d / 'complex' / 'reference_clean.pdb',
             'interface': d / 'complex' / f'interface_{IFACE_CUTOFF:g}A.txt'}
    if force or not all(p.is_file() for p in files.values()):
        for sub in ('r_u', 'r_b', 'l_u', 'l_b', 'r_u_merge', 'r_b_merge', 'complex'):
            (d / sub).mkdir(parents=True, exist_ok=True)
        with fu.change_dir(str(d)):
            for state, out in (('u', files['antibody']), ('b', d / 'r_b' / 'reference_antibody.pdb')):
                prepare_antibody(str(src[f'r_{state}']), str(out), ','.join(_chain_order(src[f'r_{state}'])),
                                 {'output_file_path': str(d / f'r_{state}_merge' / 'merged.pdb')}, {},
                                 model=None)
            for state, out in (('u', files['antigen']), ('b', d / 'l_b' / 'reference_antigen.pdb')):
                prepare_antigen(str(src[f'l_{state}']), str(out), ','.join(_chain_order(src[f'l_{state}'])),
                                model=None)
            merge_structures([str(d / 'r_b' / 'reference_antibody.pdb'), str(d / 'l_b' / 'reference_antigen.pdb')],
                             str(files['reference']))
            interface_residues(str(files['reference']), str(files['interface']))
    return {k: str(v) for k, v in files.items()}


def _chain_order(pdb_path):
    return list(dict.fromkeys(line[21] for line in Path(pdb_path).read_text().splitlines()
                              if line.startswith('ATOM')))


def fetch_benchmark(codes):
    """Extract the bound and unbound files of `codes` from the benchmark archive, once."""
    wanted = {f'{c}_{s}_{b}.pdb' for c in codes for s in 'lr' for b in 'ub'}
    missing = {n for n in wanted if not (BENCHMARK_DIR / n).is_file()}
    if not missing:
        return
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    archive = BENCHMARK_DIR / 'benchmark5.5.tgz'
    if not archive.is_file():
        urllib.request.urlretrieve(BENCHMARK_URL, archive)
    with tarfile.open(archive) as tar:
        for member in tar:
            if Path(member.name).name in missing and member.isfile():
                (BENCHMARK_DIR / Path(member.name).name).write_bytes(tar.extractfile(member).read())


def interface_residues(reference_pdb, output_txt):
    """{chain: [resid]} within IFACE_CUTOFF of the other chain, by haddock_interface.

    The reference is the fused complex of step1_6, so chain A is the antibody and chain B the
    antigen and the report lists the residues of each side near the other one.
    """
    if not Path(output_txt).is_file():
        Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
        haddock_interface(input_pdb_path=str(reference_pdb), output_txt_path=str(output_txt),
                          properties={'cutoff': IFACE_CUTOFF, 'restart': False})
    return utils.read_interface(output_txt)


def chain_types(idx):
    """{chain: ANARCII chain type} of the unbound antibody of case `idx` (H, K or L)."""
    c = case_config(idx)
    return anarcii_chain_types(c['paths']['step2_19_anarcii'], c['prop']['step2_19_anarcii'])


def anarcii_chain_types(paths, properties, force=False):
    """Run step2_19 (ANARCII on the repaired antibody) and return {chain: chain type}.

    The anarcii() function of the block returns only an exit code; the class keeps the
    'chain_type' of every chain, which is what tells the heavy chain from the light one here (the
    chain letters do not: 3HI6 uses X/Y, 4GXU M/N). It is cached next to the numbered file.
    """
    cache = Path(paths['output_pdb_path']).with_suffix('.chain_types.json')
    if force or not cache.is_file() or not Path(paths['output_pdb_path']).is_file():
        bb = Anarcii(**paths, properties=dict(properties, restart=False))
        bb.launch()
        cache.write_text(json.dumps({chain: entry['chain_type']
                                     for (_model, chain), entry in bb.results.items()}))
    return json.loads(cache.read_text())


# ============================================================================
# Residue correspondence
# ============================================================================

def read_ca(pdb_path, chain):
    """{resid: Ca coordinates} of `chain` in a prepared file (unique resids, no insertion codes)."""
    out = {}
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[21] == chain:
            out[int(line[22:26])] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return out


def residue_map(reference_pdb, target_pdb, chain, resids):
    """{reference resid: target resid} of `resids`, through utils.map_contact_residues.

    It is how stage 1 carries the reference interface onto the docking input: a global alignment
    of the observed sequences. Residues without a counterpart are left out, and so are the ones
    equally scoring alignments pair differently (a gap next to repeated residues).
    """
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        _, _, contact_map = utils.map_contact_residues(
            str(reference_pdb), str(target_pdb), chain, ', '.join(map(str, resids)),
            drop_ambiguous=True)
    return {r: t for r, t in contact_map.items() if t is not None}


def chain_slots(merged_pdb):
    """{resid of the fused chain A: position of its original chain} from the merged file of step1_1/1_4.

    The fused chain hides where one antibody chain ends and the next begins, and a global
    alignment across that junction may pair the last residue of one chain with the first of the
    next (4DN4: the light chain's C-terminal Gly with the heavy chain's first Val, 55 A apart).
    The merged file holds the same residues in the same order with their original chain IDs.
    """
    keys = []
    for line in Path(merged_pdb).read_text().splitlines():
        if line.startswith('ATOM') and (not keys or keys[-1] != (line[21], line[22:27])):
            keys.append((line[21], line[22:27]))
    order = list(dict.fromkeys(chain for chain, _ in keys))
    return {i: order.index(chain) for i, (chain, _) in enumerate(keys, 1)}


def modelled_residues(entry_pdb, prepared_pdb, chain):
    """Resids of `prepared_pdb` built by the repair rather than observed in the downloaded entry.

    The repair keeps the coordinates of the observed residues (fix_known), so a residue is
    observed when its Ca sits where a Ca of the entry does, in any model.
    """
    observed = {tuple(np.round([float(line[30:38]), float(line[38:46]), float(line[46:54])], 2))
                for line in Path(entry_pdb).read_text().splitlines()
                if line.startswith(('ATOM', 'HETATM')) and line[12:16].strip() == 'CA'}
    return {resid for resid, xyz in read_ca(prepared_pdb, chain).items()
            if tuple(np.round(xyz, 2)) not in observed}


def case_modelled(files):
    """{chain: modelled resids} of the prepared unbound antibody (A) and antigen (B)."""
    return {'A': modelled_residues(files['antibody_entry'], files['antibody'], 'A'),
            'B': modelled_residues(files['antigen_entry'], files['antigen'], 'B')}


# ============================================================================
# Superposition
# ============================================================================

def fit(x, y):
    """Rotation and translation taking `y` onto `x` (Kabsch), as y @ rot + tran."""
    xc, yc = x.mean(0), y.mean(0)
    u, _, vt = np.linalg.svd((y - yc).T @ (x - xc))
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1, 1, d]) @ vt
    return rot, xc - yc @ rot


def deviations(x, y, frame=None):
    """Per-atom distance between `x` and `y` after fitting `y` on `x`, or in a given frame."""
    rot, tran = frame if frame is not None else fit(x, y)
    return np.linalg.norm(x - (y @ rot + tran), axis=1)


def rmsd(dev):
    return float(np.sqrt(np.mean(np.square(dev)))) if len(dev) else np.nan


def irmsd(files, modelled=None):
    """I-RMSD of one case as the benchmark defines it, and its two sides.

    The interface is the residues of the bound complex within IFACE_CUTOFF of the partner. Each
    unbound partner is fitted on the Ca of its own interface residues, as the benchmark
    superposes 'the unbound component proteins onto their bound forms', and the deviations of
    both are pooled. `modelled` = case_modelled() counts the interface residues the repair built.
    """
    interface = utils.read_interface(files['interface'])
    out, pooled = {}, []
    for chain, side, target in (('A', 'ab', files['antibody']), ('B', 'ag', files['antigen'])):
        pairs = residue_map(files['reference'], target, chain, interface[chain])
        ref_ca, unb_ca = read_ca(files['reference'], chain), read_ca(target, chain)
        dev = deviations(np.array([ref_ca[r] for r in pairs]), np.array([unb_ca[u] for u in pairs.values()]))
        pooled.append(dev)
        out.update({f'irmsd_{side}': rmsd(dev), f'n_iface_{side}': len(interface[chain]),
                    f'n_iface_{side}_unmapped': len(interface[chain]) - len(pairs)})
        if modelled is not None:
            out[f'n_iface_{side}_modelled'] = len(set(pairs.values()) & modelled[chain])
    out['irmsd'] = rmsd(np.concatenate(pooled))
    return out


def interface_check(reference_pdb, interface):
    """Residues haddock_interface and an any-atom MDAnalysis search disagree on, per chain."""
    u = mda.Universe(str(reference_pdb))
    out = {}
    for chain, other in (('A', 'B'), ('B', 'A')):
        near = u.select_atoms(f'chainID {chain} and around {IFACE_CUTOFF} chainID {other}')
        out[chain] = sorted(set(near.residues.resids) ^ set(interface[chain]))
    return out


# ============================================================================
# Antibody regions
# ============================================================================

def antibody_regions(files, chain_types):
    """Regions of the prepared unbound antibody (chain A) as sets of resids, plus per-residue labels.

    The CDRs and the framework are cdr.py's IMGT ranges resolved onto the docking input by
    cdr.imgt_region_ri, as stage 2 resolves them onto the MD system. The prepared antibody is
    renumbered from 1, so its residue indices are its resids. Each CDR is split between the heavy
    and the light chain by the chain order imgt_region_ri matched and ANARCII's chain types.
    Returns the regions, {resid: 'H' | 'L'} and {resid: (IMGT position, residue name)}.
    """
    anarcii = mda.Universe(files['anarcii'])
    chains = {c: anarcii.select_atoms(f'segid {c}').residues
              for c in dict.fromkeys(anarcii.select_atoms('protein').atoms.segids)}
    framework = [(lo, min(hi, IMGT_LAST)) for lo, hi in cdr.FRAMEWORK_RANGES]
    regions, order = {}, None
    for name, ranges in (('framework', framework), ('cdr1', [cdr.CDR_RANGES[0]]),
                         ('cdr2', [cdr.CDR_RANGES[1]]), ('cdr3', [cdr.CDR_RANGES[2]])):
        ri, order, _ = cdr.imgt_region_ri(files['antibody'], files['anarcii'], ranges)
        regions[name] = set(ri)
    # imgt_region_ri checked that the chains in this order are the prepared sequence, residue by
    # residue, so the n-th residue of the order is resid n
    ordered = [(c, r) for c in order for r in chains[c]]
    chain_of = {i: 'H' if chain_types[c] == 'H' else 'L' for i, (c, _) in enumerate(ordered, 1)}
    imgt = {i: (f'{r.resid}{r.icode}'.strip(), r.resname) for i, (_, r) in enumerate(ordered, 1)}
    for i in (1, 2, 3):
        for label in 'HL':
            regions[f'{label}{i}'] = {r for r in regions[f'cdr{i}'] if chain_of[r] == label}
    regions['cdr_all'] = regions.pop('cdr1') | regions.pop('cdr2') | regions.pop('cdr3')
    regions['fv'] = regions['framework'] | regions['cdr_all']
    return regions, chain_of, imgt


def antibody_rmsds(files, chain_types, modelled=frozenset()):
    """Ca RMSD of every extent x scheme of one case, and the per-residue deviations.

    The whole bound antibody is paired with the unbound one by residue_map, keeping the pairs
    that join the same chain (chain_slots); the extents are defined on the unbound side (its IMGT
    regions) and the bound interface. Rows with a fit set smaller than MIN_FIT_RES are left NaN.
    Returns {'rows', 'residues', 'extents', 'pairs', 'frames', 'n_cross_chain'}, 'pairs' being
    {unbound resid: bound resid}.
    """
    ref_ca, unb_ca = read_ca(files['reference'], 'A'), read_ca(files['antibody'], 'A')
    ref_slot, unb_slot = chain_slots(files['reference_antibody_merged']), chain_slots(files['antibody_merged'])
    aligned = residue_map(files['reference'], files['antibody'], 'A', sorted(ref_ca))
    pairs = {u: r for r, u in aligned.items() if ref_slot[r] == unb_slot[u]}
    regions, chain_of, imgt = antibody_regions(files, chain_types)
    iface = set(utils.read_interface(files['interface'])['A'])
    extents = {name: sorted(u for u in regions[name] if u in pairs) for name in regions}
    extents['full'] = sorted(pairs)
    extents['interface'] = sorted(u for u, r in pairs.items() if r in iface)

    def coords(unb_resids):
        return (np.array([ref_ca[pairs[u]] for u in unb_resids]).reshape(-1, 3),
                np.array([unb_ca[u] for u in unb_resids]).reshape(-1, 3))

    frames = {scheme: fit(*coords(extents[ext])) for scheme, ext in SCHEMES.items() if ext}
    rows = []
    for extent in EXTENTS:
        x, y = coords(extents[extent])
        for scheme, fit_extent in SCHEMES.items():
            ok = len(x) >= (MIN_FIT_RES if fit_extent is None else 1)
            rows.append({'extent': extent, 'scheme': scheme, 'n_res': len(x),
                         'n_modelled': len(set(extents[extent]) & set(modelled)),
                         'rmsd': rmsd(deviations(x, y, frames.get(scheme))) if ok else np.nan})

    region_of = {u: name for name in ('framework', *CDRS) for u in regions[name]}
    dev = deviations(*coords(extents['full']), frames['fv_fw'])
    residues = [{'resid_unbound': u, 'resid_bound': pairs[u], 'chain_type': chain_of.get(u, ''),
                 'imgt': imgt[u][0] if u in regions['fv'] else '', 'resname': imgt[u][1],
                 'region': region_of.get(u, 'const'), 'in_interface': pairs[u] in iface,
                 'modelled': u in modelled, 'dev_ca_fv_fw': float(d)}
                for u, d in zip(extents['full'], dev)]
    return {'rows': rows, 'residues': residues, 'extents': extents, 'pairs': pairs, 'frames': frames,
            'n_cross_chain': len(aligned) - len(pairs)}


def chain_lengths(pdb_path, chains, model=None):
    """Residues of every chain of `chains`, in that order, as prepare_antigen keeps them in chain B.

    It extracts the ATOM residues of the requested model, or of the whole entry with the MODELs of an
    assembly joined and named by utils.merge_models, chain by chain in the listed order.
    """
    text = Path(pdb_path).read_text()
    if model is None:
        text = utils.merge_models(text, chains)
    elif 'MODEL' in text:
        text = text.split(f'MODEL{int(model):>9d}', 1)[1].split('ENDMDL', 1)[0]
    residues = {c: set() for c in chains}
    for line in text.splitlines():
        if line.startswith('ATOM') and line[21] in residues:
            residues[line[21]].add(line[22:27])
    return [len(residues[c]) for c in chains]


def antigen_rmsd(idx, files, modelled=frozenset()):
    """Ca RMSD of the whole unbound antigen against the bound one, fitted on itself, and how many residues.

    The fused chain B hides where one antigen chain ends and the next begins, and the alignment can
    pair across that junction (3HMX: the last residue of the unbound chain A with one of the bound
    chain B, 39.5 A apart), so pairs are kept only within the same chain, and residues the repair built
    are left out.
    """
    ids = case_config(idx)['ids']
    slots = {}
    for side, pdb, chains, model in (('reference', files['reference_entry'], ids['reference']['antigen_chains'],
                                      ids['reference']['model']),
                                     ('antigen', files['antigen_fixed'], ids['antigen']['chains'], None)):
        lengths = chain_lengths(pdb, chains.split(','), model)
        ca = read_ca(files[side], 'B')
        if sum(lengths) != len(ca):
            raise ValueError(f'case {idx}: chains {chains} of {pdb} hold {sum(lengths)} residues, '
                             f'chain B of {files[side]} {len(ca)}')
        slots[side] = dict(zip(sorted(ca), np.repeat(np.arange(len(lengths)), lengths)))
    ref, unb = read_ca(files['reference'], 'B'), read_ca(files['antigen'], 'B')
    pairs = {r: u for r, u in residue_map(files['reference'], files['antigen'], 'B', sorted(ref)).items()
             if slots['reference'][r] == slots['antigen'][u] and u not in modelled}
    dev = deviations(np.array([ref[r] for r in pairs]), np.array([unb[u] for u in pairs.values()]))
    return rmsd(dev), len(pairs)


def write_superposed(files, frame, output_pdb):
    """The prepared unbound antibody moved into the bound frame by `frame` (fit of its Ca)."""
    u = mda.Universe(files['antibody'])
    rot, tran = frame
    u.atoms.positions = u.atoms.positions @ rot + tran
    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        u.atoms.write(str(output_pdb))
    return str(output_pdb)


def ngl_selection(resids, chain='A'):
    """NGL selection of `resids` of one chain."""
    return ' or '.join(f'{r}:{chain}' for r in resids)


# ============================================================================
# Per-case tables
# ============================================================================

def load_cases(indices=None):
    """Prepare every case and its benchmark control, then tabulate them.

    Returns (files, bench_files, modelled, cases): case_files, prepare_benchmark_case and
    case_modelled per case index, and one row of case_info per case with the number of residues
    the repair built in the antibody and the antigen.
    """
    indices = range(len(COMPLEXES)) if indices is None else indices
    files, bench_files, modelled = {}, {}, {}
    for idx in indices:
        files[idx] = prepare_case(idx)
        bench_files[idx] = prepare_benchmark_case(idx)
        modelled[idx] = case_modelled(files[idx])
    cases = pd.DataFrame([{**case_info(idx), 'antibody_modelled': len(modelled[idx]['A']),
                           'antigen_modelled': len(modelled[idx]['B'])}
                          for idx in files]).set_index('case_idx')
    return files, bench_files, modelled, cases


def interface_table(files):
    """Interface size per case, and how many residues haddock_interface and MDAnalysis disagree on."""
    rows = []
    for idx, f in files.items():
        interface = utils.read_interface(f['interface'])
        disagree = interface_check(f['reference'], interface)
        rows.append({'case_idx': idx, 'antibody': len(interface['A']), 'antigen': len(interface['B']),
                     'disagree_with_mdanalysis': sum(map(len, disagree.values()))})
    iface = pd.DataFrame(rows).set_index('case_idx')
    assert iface.disagree_with_mdanalysis.sum() == 0, 'haddock_interface is not the any-atom 10 A interface'
    return iface


IRMSD_COLUMNS = ['ref_code', 'unb_code', 'benchmark_irmsd', 'bench_irmsd', 'bench_irmsd_ab', 'bench_irmsd_ag',
                 'wf_irmsd', 'wf_irmsd_ab', 'wf_irmsd_ag']


def irmsd_table(cases, files, bench_files, modelled):
    """I-RMSD per case of the benchmark files ('bench_*') and of the workflow structures ('wf_*').

    'wf_antigen' is the whole workflow antigen against the bound one (antigen_rmsd), over 'wf_n_antigen'
    residues.
    """
    rows = []
    for idx in files:
        bench, ours = irmsd(bench_files[idx]), irmsd(files[idx], modelled[idx])
        antigen, n_antigen = antigen_rmsd(idx, files[idx], modelled[idx]['B'])
        rows.append({'case_idx': idx, **{f'bench_{k}': v for k, v in bench.items()},
                     **{f'wf_{k}': v for k, v in ours.items()},
                     'wf_antigen': antigen, 'wf_n_antigen': n_antigen})
    irm = cases[['ref_code', 'unb_code', 'difficulty', 'benchmark_irmsd']].join(
        pd.DataFrame(rows).set_index('case_idx'))
    irm.to_csv(RES / 'ab_rmsds_irmsd.csv')
    return irm


def report_irmsd(irm, tol=0.015):
    """How the benchmark files compare with the published column, and the workflow with them."""
    d_bench = irm.bench_irmsd - irm.benchmark_irmsd
    d_wf = irm.wf_irmsd - irm.bench_irmsd
    n = len(irm)
    print(f'benchmark files - published: |diff| <= 0.01 A in {(d_bench.abs() <= tol).sum()}/{n}, '
          f'largest {d_bench.abs().max():.2f} A ({irm.ref_code[d_bench.abs().idxmax()]})')
    print(f'workflow - benchmark files : |diff| <= 0.01 A in {(d_wf.abs() <= tol).sum()}/{n}; differ in')
    for idx in d_wf[d_wf.abs() > tol].index:
        r = irm.loc[idx]
        print(f'    {r.ref_code}/{r.unb_code}: {r.wf_irmsd:.2f} vs {r.bench_irmsd:.2f} A '
              f'(antibody side {r.wf_irmsd_ab:.2f} vs {r.bench_irmsd_ab:.2f}, '
              f'antigen side {r.wf_irmsd_ag:.2f} vs {r.bench_irmsd_ag:.2f})')
    print('interface residues without an unbound counterpart (workflow):',
          int(irm.wf_n_iface_ab_unmapped.sum() + irm.wf_n_iface_ag_unmapped.sum()))
    print('interface residues built by the repair (workflow)          :',
          int(irm.wf_n_iface_ab_modelled.sum() + irm.wf_n_iface_ag_modelled.sum()))
    assert d_bench.abs().max() < 0.05, 'the benchmark files do not reproduce the published I-RMSD'


def report_sides(irm):
    """Which partner carries the I-RMSD, on the benchmark files."""
    side = np.where(irm.bench_irmsd_ag > irm.bench_irmsd_ab, 'antigen', 'antibody')
    print('larger side per case:', dict(pd.Series(side).value_counts()))
    for diff, grp in irm.groupby('difficulty'):
        print(f'{diff:6s}: median antibody side {grp.bench_irmsd_ab.median():.2f} A, '
              f'antigen side {grp.bench_irmsd_ag.median():.2f} A')


def antibody_tables(files, modelled, cases):
    """antibody_rmsds of every case, as a long table and a per-residue table (both saved).

    Returns (df, dfres, ab): one row per case x extent x scheme, one row per paired residue, and
    the antibody_rmsds output per case.
    """
    records, residue_records, ab = [], [], {}
    for idx, f in files.items():
        ab[idx] = antibody_rmsds(f, chain_types(idx), modelled[idx]['A'])
        records += [{'case_idx': idx, **row} for row in ab[idx]['rows']]
        residue_records += [{'case_idx': idx, 'ref_code': cases.ref_code[idx], **row}
                            for row in ab[idx]['residues']]
    df, dfres = pd.DataFrame(records), pd.DataFrame(residue_records)
    df.to_csv(RES / 'ab_rmsds_long.csv', index=False)
    dfres.to_csv(RES / 'ab_rmsds_residues.csv', index=False)
    return df, dfres, ab


def report_pairing(files, df, ab):
    """What the pairing of the whole antibody left out; returns the residues per extent and case."""
    print(f'{len(df)} RMSD rows ({len(EXTENTS)} extents x {len(SCHEMES)} schemes x {len(files)} cases)')
    n_bound = {idx: len(read_ca(f['reference'], 'A')) for idx, f in files.items()}
    print('bound antibody residues left unpaired (no counterpart, or ambiguous):',
          {idx: n_bound[idx] - len(ab[idx]['pairs']) for idx in files})
    print('pairs dropped for crossing the light/heavy junction:',
          {idx: a['n_cross_chain'] for idx, a in ab.items() if a['n_cross_chain']})
    full = df[(df.extent == 'full') & (df.scheme == 'self')].set_index('case_idx')
    print('paired residues the repair built:', {idx: n for idx, n in full.n_modelled.items() if n})
    return df[df.scheme == 'self'].pivot_table(index='case_idx', columns='extent', values='n_res')[EXTENTS].T


# Per-case headline numbers: column -> (extent, scheme)
HEADLINE = {
    'framework': ('framework', 'self'),
    'fv': ('fv', 'self'),
    'full': ('full', 'self'),
    'fv_in_full': ('fv', 'full'),
    'interface_ab': ('interface', 'self'),
    'cdr_self': ('cdr_all', 'self'),
    'cdr_fw': ('cdr_all', 'fv_fw'),
    'H3_fw': ('H3', 'fv_fw'),
}


def pick(df, extent, scheme):
    """RMSD of one extent and scheme from the long table, indexed by case."""
    sel = df[(df.extent == extent) & (df.scheme == scheme)]
    return sel.set_index('case_idx')['rmsd']


def summary_table(irm, df):
    """One row per case: the I-RMSDs and the HEADLINE RMSDs, sorted by published I-RMSD (saved)."""
    summary = irm[['ref_code', 'unb_code', 'difficulty', 'benchmark_irmsd', 'bench_irmsd', 'wf_irmsd',
                   'wf_irmsd_ab', 'wf_irmsd_ag', 'wf_antigen']].copy()
    for name, (extent, scheme) in HEADLINE.items():
        summary[name] = pick(df, extent, scheme)
    summary = summary.sort_values('benchmark_irmsd')
    summary.to_csv(RES / 'ab_rmsds_summary.csv')
    return summary


def run_checks(files, df, summary, idx=13):
    """Identity, lower bound, nesting and agreement of the two pairings; raises on a failure."""
    f = files[idx]
    identity = irmsd({**f, 'antibody': f['reference'], 'antigen': f['reference']})
    assert identity['irmsd'] < 1e-6, identity
    print(f"1. identity: I-RMSD of the bound complex against itself = {identity['irmsd']:.1e} A")

    wide = df.pivot_table(index=['case_idx', 'extent'], columns='scheme', values='rmsd')
    below = {s: int((wide[s] < wide['self'] - 1e-9).sum()) for s in ('fv_fw', 'full')}
    assert not any(below.values()), below
    print(f'2. lower bound: no fixed-frame value below its self value ({len(wide.dropna())} rows)')

    sizes = df[df.scheme == 'fv_fw'].pivot_table(index='case_idx', columns='extent', values='n_res')
    nest = {'framework + cdr_all == fv': (sizes.framework + sizes.cdr_all == sizes.fv).all(),
            'H1 + ... + L3 == cdr_all': (sizes[CDRS].sum(axis=1) == sizes.cdr_all).all(),
            'fv <= full': (sizes.fv <= sizes.full).all()}
    assert all(nest.values()), nest
    print('3. nesting:', ', '.join(nest))

    gap = (summary.interface_ab - summary.wf_irmsd_ab).abs()
    assert gap.max() < 0.01, gap
    print(f'4. antibody side of the I-RMSD vs interface extent: largest difference {gap.max():.1e} A')


def report_antigen(summary):
    """The antigen, the benchmark's ligand: the whole of it against its side of the interface (figure 3, below)."""
    for key, label in (('wf_antigen', 'whole antigen (self)        '), ('wf_irmsd_ag', 'interface, antigen side (self)')):
        print(f'{label}: {summary[key].min():.2f} - {summary[key].max():.2f} A (median {summary[key].median():.2f})')
    more = summary.wf_antigen > summary.wf_irmsd_ag
    print(f'the whole antigen differs more than its interface side in {int(more.sum())}/{len(summary)} cases; largest: '
          + ', '.join(f'{r.ref_code} {r.wf_antigen:.2f} A' for _, r in summary.nlargest(3, 'wf_antigen').iterrows()))


def report_cdr_vs_framework(summary):
    ratio = summary.cdr_fw / summary.framework
    print('CDR displacement / framework RMSD, per case:')
    print(f'  median {ratio.median():.2f}x, range {ratio.min():.2f}-{ratio.max():.2f}x')
    print(f'  CDRs move more than the framework in {int((ratio > 1).sum())}/{len(ratio)} cases')
    print(f'  Wilcoxon signed-rank (CDR vs framework): p = '
          f'{stats.wilcoxon(summary.cdr_fw, summary.framework).pvalue:.2g}')


def report_per_cdr(df, cases):
    per_cdr = pd.DataFrame({c: pick(df, c, 'fv_fw') for c in CDRS})
    print('median displacement in the framework frame, per CDR (A):')
    print(per_cdr.median().round(2).to_string())
    print(f'\nCDR-H3 is the largest of the six in {int((per_cdr.idxmax(axis=1) == "H3").sum())}/'
          f'{len(per_cdr)} cases')
    top = per_cdr.stack().nlargest(4)
    print('largest single-loop displacements:',
          ', '.join(f'{cases.ref_code[i]} {c} {v:.2f} A' for (i, c), v in top.items()))


def report_elbow(summary):
    for key, label in (('full', 'whole-Fab RMSD (full/self)   '), ('fv', 'whole-Fv RMSD (fv/self)      '),
                       ('fv_in_full', 'Fv residual in a whole-Fab fit')):
        print(f'{label}: {summary[key].min():.2f} - {summary[key].max():.2f} A '
              f'(median {summary[key].median():.2f})')
    print(f'the whole-Fab fit inflates the Fv RMSD by more than 50% in '
          f'{int((summary.fv_in_full > 1.5 * summary.fv).sum())}/{len(summary)} cases')


# ============================================================================
# The largest CDR-H3 displacement
# ============================================================================

def worst_h3(cases, files, df, dfres, ab):
    """The case whose CDR-H3 moves most in the framework frame, its loop residue by residue, and
    its unbound antibody written out superposed on the bound one by the Fv framework Ca."""
    idx = int(pick(df, 'H3', 'fv_fw').idxmax())
    h3 = dfres[(dfres.case_idx == idx) & (dfres.region == 'H3')].sort_values('resid_unbound')
    out = OUT / f'worst_H3_{cases.unb_code[idx]}_on_{cases.ref_code[idx]}.pdb'
    return {'idx': idx, 'case': cases.loc[idx], 'files': files[idx], 'ab': ab[idx], 'h3': h3,
            'tip': h3.loc[h3.dev_ca_fv_fw.idxmax()], 'rmsd': pick(df, 'H3', 'fv_fw')[idx],
            'superposed': write_superposed(files[idx], ab[idx]['frames']['fv_fw'], out)}


def report_worst_h3(w):
    case, h3, tip = w['case'], w['h3'], w['tip']
    print(f"largest CDR-H3 displacement: case {w['idx']}, {case.reference} (bound) vs {case.antibody} "
          f"(unbound) = {w['rmsd']:.2f} A Ca RMSD")
    print(f"fitted on {len(w['ab']['extents']['framework'])} framework Ca (no CDR atom takes part in the fit)")
    print('superposed unbound antibody:', w['superposed'])
    print(f'the loop tip moves furthest: IMGT {tip.imgt} {tip.resname} at {tip.dev_ca_fv_fw:.2f} A; its anchors '
          f'(IMGT {h3.imgt.iloc[0]} and {h3.imgt.iloc[-1]}) stay within '
          f'{max(h3.dev_ca_fv_fw.iloc[0], h3.dev_ca_fv_fw.iloc[-1]):.2f} A')
    print(f'{int(h3.in_interface.sum())}/{len(h3)} loop residues are in the 10 A interface')
    return (h3[['imgt', 'resname', 'resid_bound', 'resid_unbound', 'in_interface', 'dev_ca_fv_fw']]
            .rename(columns={'dev_ca_fv_fw': 'Ca displacement (A)'}).round(2).set_index('imgt'))


def h3_view(w):
    """NGL view of the loop: bound Fv in blue, unbound Fv superposed in grey, antigen faint;
    bound CDR-H3 green and unbound orange, the loop tip's Ca as a sphere."""
    import nglview as nv

    case, h3, tip, ab = w['case'], w['h3'], w['tip'], w['ab']
    sel_bound, sel_unbound = ngl_selection(h3.resid_bound), ngl_selection(h3.resid_unbound)
    fv = ab['extents']['fv']
    view = nv.NGLWidget()
    bound = view.add_component(w['files']['reference'], name=f'{case.ref_code} bound')
    unbound = view.add_component(w['superposed'], name=f'{case.unb_code} unbound (superposed)')
    view.clear_representations(component=0)
    view.clear_representations(component=1)
    bound.add_cartoon(selection=ngl_selection([ab['pairs'][u] for u in fv]), color='#6da7ec', opacity=0.3)
    bound.add_cartoon(selection=':B', color='#b0b0b0', opacity=0.15)
    unbound.add_cartoon(selection=ngl_selection(fv), color='#9a9a9a', opacity=0.3)
    for comp, sel, color in ((bound, sel_bound, '#1baf7a'), (unbound, sel_unbound, '#eb6834')):
        comp.add_cartoon(selection=sel, color=color, aspectRatio=2.0, scale=1.4)
        comp.add_ball_and_stick(selection=sel, color=color, aspectRatio=1.6)
    bound.add_spacefill(selection=f'{tip.resid_bound}:A and .CA', color='#1baf7a', radius=0.9)
    unbound.add_spacefill(selection=f'{tip.resid_unbound}:A and .CA', color='#eb6834', radius=0.9)
    view.add_label(selection=f'{tip.resid_bound}:A and .CA', component=0, labelType='format',
                   labelFormat=f'{seq1(tip.resname)}{tip.imgt} ({tip.dev_ca_fv_fw:.1f} Å)',
                   color='#111111', radius=2.2)
    view.center(selection=sel_bound, component=0)
    view._set_size('100%', '600px')
    return view


# ============================================================================
# Docking
# ============================================================================

# The cluster runs copied back from the HPC (array/README.md), one folder per case
HPC_RESULTS = ROOT / 'output' / 'hpc_results'
DOCKING_RUNS = {'baseline': 'step1_12_haddock3_run', 'md': 'step2_26_haddock3_run',
                'awh': 'step3_30_haddock3_run'}
PROTOCOL_LABEL = {'baseline': 'baseline docking', 'md': 'free-MD CDR ensemble', 'awh': 'AWH CDR ensemble'}
# caprieval after emref: every refined model of the run, 200 per run
CAPRI_TABLE = Path('haddock_output') / 'run' / '08_caprieval' / 'capri_ss.tsv'
# caprieval after seletopclusts; stage 3 simulates its cluster_1_model_1 (utils.haddock_best_model)
CLUSTER_TABLE = Path('haddock_output') / 'run' / '11_caprieval' / 'capri_ss.tsv'
# DockQ quality classes (Basu & Wallner 2016): acceptable, medium
ACCEPTABLE, MEDIUM = 0.23, 0.49
TOP = 10
# The cluster runs of these cases in HPC_RESULTS were scored against a reference whose antigen
# protomers came in file order (prepare_antigen before the fix of finding 2), which caps their
# DockQ. They are to be run again: empty this once their new tables are copied in
FILE_ORDER_REFERENCE = ('4FQI', '4GXU')
# DockQ is the mean of fnat and of the i-RMSD and L-RMSD scaled by these (Basu & Wallner 2016)
DOCKQ_SCALE = {'irmsd': 1.5, 'lrmsd': 8.5}
# A model with the right interface: the i-RMSD of a medium-quality CAPRI model, and the fewest such
# models whose median L-RMSD is read
RIGHT_INTERFACE, MIN_RIGHT = 2.0, 5
# The CDR-loop clusters each ensemble docked, one MODEL per cluster, copied back with the tables
ENSEMBLE_PDB = {'md': Path('step2_23_gmx_cluster') / 'antibody_clusters.pdb',
                'awh': Path('step3_27_gmx_cluster') / 'antibody_clusters.pdb'}
# What a docked antibody structure is measured by, as (extent, fitted extent) of antibody_rmsds: the
# Fv in a whole-Fab fit, which is the superposition caprieval's L-RMSD makes, and the CDRs in the
# framework frame
STRUCTURE_RMSDS = {'fv_in_full': ('fv', 'full'), 'cdr_fw': ('cdr_all', 'framework'),
                   'H3_fw': ('H3', 'framework'), 'L3_fw': ('L3', 'framework')}
# CHARMM names histidine by its protonation state, calls Ile CD1 CD and the C-terminal oxygens OT1/OT2
CHARMM_RESNAMES = {'HSD': 'HIS', 'HSE': 'HIS', 'HSP': 'HIS'}
CHARMM_ATOMS = {('ILE', 'CD'): 'CD1'}
BACKBONE = {'N', 'CA', 'C', 'O', 'OXT', 'OT1', 'OT2'}
# The paratope: the antibody residues with a heavy atom this close to the antigen in the bound complex
PARATOPE_CUTOFF = 4.5
# Its side chains, each residue on its own backbone (side_chain_rmsds); lower is closer to bound in both
SIDE_CHAIN_RMSDS = ('paratope_sc', 'paratope_chi1_off')
CHI1_TOLERANCE = 40
CHI1_GAMMA = {'SER': 'OG', 'THR': 'OG1', 'CYS': 'SG', 'VAL': 'CG1', 'ILE': 'CG1'}  # CG in the others
# Atom names a 180-degree flip of the side chain exchanges
SYMMETRIC = {'PHE': (('CD1', 'CD2'), ('CE1', 'CE2')), 'TYR': (('CD1', 'CD2'), ('CE1', 'CE2')),
             'ASP': (('OD1', 'OD2'),), 'GLU': (('OE1', 'OE2'),), 'ARG': (('NH1', 'NH2'),)}


def docking_table(cases):
    """DockQ of the docked models per case and protocol, from the runs' own caprieval.

    'acceptable' is the fraction of the refined models with DockQ >= ACCEPTABLE, what the protocol
    samples; 'top' counts the acceptable ones among the TOP best HADDOCK scores, what scoring picks.
    'start_dockq' is the DockQ of the run's best cluster model, which stage 3 starts the AWH from
    when the run is the baseline one.
    """
    rows = []
    for idx in cases.index:
        for protocol, run in DOCKING_RUNS.items():
            table = HPC_RESULTS / f'case_{idx}' / run / CAPRI_TABLE
            if not table.is_file():
                continue
            d = pd.read_csv(table, sep='\t').sort_values('score')
            best = d.loc[d.dockq.idxmax()]
            clusters = pd.read_csv(HPC_RESULTS / f'case_{idx}' / run / CLUSTER_TABLE, sep='\t')
            start = clusters[clusters.model.str.endswith('cluster_1_model_1.pdb')].dockq
            rows.append({'case_idx': idx, 'ref_code': cases.ref_code[idx], 'protocol': protocol,
                         'n_models': len(d), 'best_dockq': best.dockq, 'acceptable': (d.dockq >= ACCEPTABLE).mean(),
                         'top': int((d.dockq.head(TOP) >= ACCEPTABLE).sum()),
                         'start_dockq': start.iloc[0] if len(start) else np.nan,
                         'best_fnat': best.fnat, 'best_irmsd': best.irmsd, 'best_lrmsd': best.lrmsd,
                         'max_fnat': d.fnat.max(), 'min_irmsd': d.irmsd.min(), 'min_lrmsd': d.lrmsd.min()})
    dock = pd.DataFrame(rows)
    dock.to_csv(RES / 'ab_rmsds_docking.csv', index=False)
    return dock


def docking_gain(dock, summary):
    """Per case: the CDR displacement next to each ensemble's change against the baseline docking."""
    wide = dock.pivot_table(index='case_idx', columns='protocol',
                            values=['best_dockq', 'acceptable', 'top', 'start_dockq'])
    out = summary[['ref_code', 'difficulty', 'cdr_fw', 'H3_fw']].copy()
    for metric in ('acceptable', 'best_dockq', 'top'):
        for protocol in DOCKING_RUNS:
            out[f'{metric}_{protocol}'] = wide[(metric, protocol)]
        for protocol in ('md', 'awh'):
            out[f'd_{metric}_{protocol}'] = wide[(metric, protocol)] - wide[(metric, 'baseline')]
    out['awh_start_dockq'] = wide[('start_dockq', 'baseline')]
    return out.sort_values('cdr_fw', ascending=False)


def perfect_pose_dockq(files, bench_files):
    """DockQ of the right answer, per case, against the workflow's reference and the benchmark's.

    The right answer is the benchmark's unbound partners where they bind: the benchmark ships its
    _u files superposed on the bound complex. It is fused as stage 1 fuses the docking input
    (antibody chain A, antigen chain B, chains in file order) and scored by haddock3's caprieval
    with its defaults, as the docking runs score their models. The workflow's reference is step1_6
    as the current code prepares it; the benchmark's bound files order their chains to pair with
    the unbound ones (ABEFCD with ABCDEF or HIJKLM in 4FQI and 4GXU), and so does prepare_antigen,
    which follows the identifier.
    """
    import logging
    from haddock.gear.yaml2cfg import read_from_yaml_config
    from haddock.modules.analysis.caprieval import DEFAULT_CONFIG
    from haddock.modules.analysis.caprieval.capri import CAPRI

    logging.getLogger('haddock').setLevel(logging.WARNING)
    params = read_from_yaml_config(DEFAULT_CONFIG)
    rows = []
    for idx, f in files.items():
        work = Path(f['interface']).parent / 'perfect_pose'
        work.mkdir(parents=True, exist_ok=True)
        pose = work / 'unbound_at_bound_position.pdb'
        pose.write_text('\n'.join(line for p in (bench_files[idx]['antibody'], bench_files[idx]['antigen'])
                                  for line in Path(p).read_text().splitlines() if line.startswith('ATOM'))
                        + '\nEND\n')
        row = {'case_idx': idx}
        for name, reference in (('workflow', f['reference']), ('benchmark', bench_files[idx]['reference'])):
            (work / name).mkdir(exist_ok=True)
            capri = CAPRI(1, pose, work / name, Path(reference), params)
            with contextlib.redirect_stdout(io.StringIO()):
                capri.run()
            row.update({f'{k}_{name}': getattr(capri, k) for k in ('fnat', 'irmsd', 'lrmsd', 'dockq')})
        rows.append(row)
    pp = pd.DataFrame(rows).set_index('case_idx')
    pp.to_csv(RES / 'ab_rmsds_perfect_pose.csv')
    return pp


def docking_models(cases):
    """Every refined model of the cluster runs, one row each: its HADDOCK score and caprieval metrics."""
    frames = []
    for idx in cases.index:
        for protocol, run in DOCKING_RUNS.items():
            table = HPC_RESULTS / f'case_{idx}' / run / CAPRI_TABLE
            if table.is_file():
                d = pd.read_csv(table, sep='\t', usecols=['model', 'score', 'fnat', 'irmsd', 'lrmsd', 'dockq'])
                frames.append(d.assign(case_idx=idx, ref_code=cases.ref_code[idx], protocol=protocol))
    return pd.concat(frames, ignore_index=True)


def dockq_terms(models):
    """Per case, what each ensemble changes in the mean of every DockQ term over its refined models.

    DockQ = (fnat + 1/(1 + (i-RMSD/1.5)^2) + 1/(1 + (L-RMSD/8.5)^2)) / 3, so the change of the mean
    DockQ is a third of the sum of the changes of its terms ('d_<term>_<protocol>'). The L-RMSD is
    caprieval's: the RMSD of the antigen (chain B) once the whole antibody (chain A, constant
    domains included) is superposed on the bound one.
    """
    terms = models.assign(fnat_term=models.fnat,
                          **{f'{m}_term': 1 / (1 + (models[m] / s) ** 2) for m, s in DOCKQ_SCALE.items()})
    names = ['dockq', 'fnat_term', 'irmsd_term', 'lrmsd_term']
    mean = terms.groupby(['case_idx', 'protocol'])[names].mean().unstack()
    out = models.groupby('case_idx').ref_code.first().to_frame()
    for protocol in ('md', 'awh'):
        for name in names:
            out[f'd_{name.removesuffix("_term")}_{protocol}'] = mean[(name, protocol)] - mean[(name, 'baseline')]
    return out.dropna(subset=['d_dockq_awh'])


def read_residues(lines, chain=None):
    """[(residue name, {heavy atom: coordinates})] of the ATOM records, in file order, CHARMM names translated.

    A residue starts at every new chain or residue number and at every backbone N: the cluster
    structures keep the entry's numbering without its insertion codes (ASP 52 then PRO 52).
    """
    residues, key = [], None
    for line in lines:
        if not line.startswith('ATOM') or (chain and line[21] != chain):
            continue
        atom, name = line[12:16].strip(), line[17:21].strip()
        name = CHARMM_RESNAMES.get(name, name)
        if atom.lstrip('0123456789').startswith('H'):
            continue
        if line[21:27] != key or (atom == 'N' and 'N' in residues[-1][1]):
            residues.append((name, {}))
            key = line[21:27]
        residues[-1][1][CHARMM_ATOMS.get((name, atom), atom)] = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return residues


def read_models(pdb_path):
    """read_residues of every MODEL of a multi-model PDB file."""
    models = []
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith('MODEL'):
            models.append([])
        elif models:
            models[-1].append(line)
    return [read_residues(lines) for lines in models]


def structure_rmsds(reference_pdb, a, ca):
    """STRUCTURE_RMSDS of one antibody structure against the bound one.

    `ca` = {unbound resid: Ca coordinates}; `a` is the antibody_rmsds output of the case, whose pairs
    and extents are used as they are.
    """
    ref = read_ca(reference_pdb, 'A')

    def coords(extent):
        resids = a['extents'][extent]
        return np.array([ref[a['pairs'][u]] for u in resids]), np.array([ca[u] for u in resids])
    return {name: rmsd(deviations(*coords(extent), fit(*coords(fitted))))
            for name, (extent, fitted) in STRUCTURE_RMSDS.items()}


def paratope(bound, antigen, pairs, unbound):
    """{bound resid: unbound resid} of the paratope residues, of the same type on both sides.

    The paratope is the bound antibody residues with a heavy atom within PARATOPE_CUTOFF of the antigen;
    `bound` and `unbound` are {resid: read_residues entry}, `antigen` the bound antigen's residues and
    `pairs` antibody_rmsds' {unbound resid: bound resid}.
    """
    tree = cKDTree(np.array([xyz for _, atoms in antigen for xyz in atoms.values()]))
    to_unbound = {r: u for u, r in pairs.items()}
    return {r: to_unbound[r] for r, (name, atoms) in bound.items()
            if r in to_unbound and unbound[to_unbound[r]][0] == name
            and tree.query(np.array(list(atoms.values())))[0].min() <= PARATOPE_CUTOFF}


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, (p2 - p1) / np.linalg.norm(p2 - p1), p3 - p2
    v, w = b0 - np.dot(b0, b1) * b1, b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def side_chain_rmsds(bound, residues, para):
    """SIDE_CHAIN_RMSDS of one antibody structure over the paratope `para` = {bound resid: resid}.

    Every residue is superposed on the bound one by its own N, Ca and C, so the loop it sits on does not
    count: 'paratope_sc' is the RMSD of the atoms past Cb, with the naming of SYMMETRIC atoms that fits
    better; 'paratope_chi1_off' the fraction of chi1 angles more than CHI1_TOLERANCE degrees off.
    """
    dev, off = [], []
    for r, u in para.items():
        name, ref = bound[r]
        atoms = residues[u][1]
        side = [a for a in ref if a not in BACKBONE and a != 'CB' and a in atoms]
        if side:
            frame = fit(np.array([ref[a] for a in ('N', 'CA', 'C')]), np.array([atoms[a] for a in ('N', 'CA', 'C')]))
            x = np.array([ref[a] for a in side])
            namings = [dict(zip(side, side))]
            if name in SYMMETRIC:
                flip = dict(sum(([(p, q), (q, p)] for p, q in SYMMETRIC[name]), []))
                namings.append({a: flip.get(a, a) for a in side})
            dev.append(min((deviations(x, np.array([atoms[naming[a]] for a in side]), frame) for naming in namings),
                           key=rmsd))
        gamma = CHI1_GAMMA.get(name, 'CG')
        if all(a in ref and a in atoms for a in ('N', 'CA', 'CB', gamma)):
            chi = [_dihedral(*(at[a] for a in ('N', 'CA', 'CB', gamma))) for at in (ref, atoms)]
            off.append(abs((chi[0] - chi[1] + 180) % 360 - 180) > CHI1_TOLERANCE)
    return {'paratope_sc': rmsd(np.concatenate(dev)), 'paratope_chi1_off': float(np.mean(off))}


def ensemble_rmsds(files, ab):
    """How far every docked antibody structure is from the bound one (saved).

    The baseline docking docks the prepared unbound antibody; the MD and AWH ensembles dock the
    cluster structures of ENSEMBLE_PDB, which come from the MD of that same antibody and so hold its
    residues in its order. That is checked residue by residue, and so is that no chain is broken
    across the periodic box, before they are measured as the unbound one is: STRUCTURE_RMSDS on the
    Ca, SIDE_CHAIN_RMSDS over the paratope. One row per structure.
    """
    rows = []
    for idx, f in files.items():
        resids = sorted(read_ca(f['antibody'], 'A'))
        unbound = dict(zip(resids, read_residues(Path(f['antibody']).read_text().splitlines(), 'A'), strict=True))
        ref_lines = Path(f['reference']).read_text().splitlines()
        bound = dict(zip(sorted(read_ca(f['reference'], 'A')), read_residues(ref_lines, 'A'), strict=True))
        para = paratope(bound, read_residues(ref_lines, 'B'), ab[idx]['pairs'], unbound)
        names = [name for name, _ in unbound.values()]
        slot = chain_slots(f['antibody_merged'])
        structures = [('unbound', 0, unbound)]
        for source, rel in ENSEMBLE_PDB.items():
            path = HPC_RESULTS / f'case_{idx}' / rel
            if not path.is_file():
                continue
            for member, model in enumerate(read_models(path), 1):
                if [name for name, _ in model] != names:
                    raise ValueError(f'{path} model {member} is not the prepared antibody of case {idx}')
                structures.append((source, member, dict(zip(resids, model))))
        for source, member, residues in structures:
            ca = {u: atoms['CA'] for u, (_, atoms) in residues.items()}
            # an MD structure has no gap of its own, a long Ca-Ca step is the periodic box
            gap = max(np.linalg.norm(ca[u] - ca[v]) for u, v in zip(resids, resids[1:]) if slot[u] == slot[v])
            if source != 'unbound' and gap > 4.5:
                raise ValueError(f'case {idx} {source} {member}: a chain is broken ({gap:.1f} A between Ca)')
            rows.append({'case_idx': idx, 'source': source, 'member': member, 'n_paratope': len(para),
                         **structure_rmsds(f['reference'], ab[idx], ca), **side_chain_rmsds(bound, residues, para)})
    ens = pd.DataFrame(rows)
    ens.to_csv(RES / 'ab_rmsds_ensembles.csv', index=False)
    return ens


def ensemble_table(ens, summary):
    """Per case: the unbound value, and the median and the closest member of each ensemble, of every measure.

    The unbound antibody, measured here as the members are, must give the summary's values.
    """
    unbound = ens[ens.source == 'unbound'].set_index('case_idx')
    measured = ['fv_in_full', 'cdr_fw', 'H3_fw']
    gap = (unbound[measured] - summary.loc[unbound.index, measured]).abs().max().max()
    assert gap < 1e-6, f'the unbound antibody measures {gap:.2g} A away from the summary'
    out = summary[['ref_code']].copy()
    out['n_paratope'] = unbound.n_paratope
    for name in (*STRUCTURE_RMSDS, *SIDE_CHAIN_RMSDS):
        stat = ens.groupby(['case_idx', 'source'])[name].agg(['median', 'min']).unstack()
        out[f'{name}_unbound'] = stat[('median', 'unbound')]
        for source in ENSEMBLE_PDB:
            out[f'{name}_{source}'] = stat[('median', source)]
            out[f'{name}_{source}_min'] = stat[('min', source)]
    return out.dropna(subset=['cdr_fw_awh'])


def lrmsd_vs_elbow(models, ens):
    """Per case and protocol: the L-RMSD of the right-interface models next to the elbow of what was docked.

    'lrmsd_right' is the median L-RMSD of the models with i-RMSD <= RIGHT_INTERFACE ('n_right' of
    them; NaN under MIN_RIGHT), 'elbow' the median Fv error in a whole-Fab fit of the structures the
    protocol docked: the unbound antibody for the baseline, the cluster members for the ensembles.
    """
    right = models[models.irmsd <= RIGHT_INTERFACE].groupby(['case_idx', 'protocol']).lrmsd.agg(['median', 'size'])
    elbow = ens.groupby(['case_idx', 'source']).fv_in_full.median()
    rows = []
    for (idx, protocol), (median, n) in right.iterrows():
        source = 'unbound' if protocol == 'baseline' else protocol
        if (idx, source) in elbow.index:
            rows.append({'case_idx': idx, 'ref_code': models.ref_code[models.case_idx == idx].iloc[0],
                         'protocol': protocol, 'n_right': int(n), 'elbow': elbow[(idx, source)],
                         'lrmsd_right': median if n >= MIN_RIGHT else np.nan})
    return pd.DataFrame(rows)


# ============================================================================
# Figures
# ============================================================================

# Categorical slots 1-2 and 1-5 of the reference data-viz palette, validated for colour-vision
# deficiency all-pairs (2 classes) and adjacent-pairs (5 bars); the 5 bars carry a legend and the
# summary table is the table view
DIFF_COLOR = {'rigid': '#2a78d6', 'medium': '#eb6834'}
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
# the hue order of the workflow's own results figure: baseline blue, free MD orange, AWH green
PROTOCOL_COLOR = dict(zip(DOCKING_RUNS, SERIES))
INK, INK_MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
# Sequential blue ramp, light -> dark, for magnitudes
BLUES = LinearSegmentedColormap.from_list(
    'blues', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])


def setup_figures():
    """Output folders and the plotting theme of the notebook."""
    for d in (RES, FIG):
        d.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings('ignore', category=UserWarning)
    sns.set_theme(style='whitegrid', context='notebook', rc={'grid.color': GRID})
    pd.set_option('display.width', 200, 'display.max_columns', 40)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches='tight')
    plt.show()


def label_points(ax, xs, ys, labels, tol=0.04, same_units=True):
    """Label the points of a scatter; points closer than `tol` of the data span share one label.

    With `same_units` (an equal-aspect plot) both axes are scaled by the larger span, otherwise each
    by its own, so that Å against percentage points does not count every x as close.
    """
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    sx, sy = np.ptp(xs) or 1.0, np.ptp(ys) or 1.0
    if same_units:
        sx = sy = max(sx, sy)
    groups = []
    for i in np.argsort(xs):
        for group in groups:
            if np.hypot((xs[i] - xs[group[0]]) / sx, (ys[i] - ys[group[0]]) / sy) < tol:
                group.append(i)
                break
        else:
            groups.append([i])
    for group in groups:
        right = max(group, key=lambda i: xs[i])
        names = dict.fromkeys(str(list(labels)[i]) for i in group)  # one case in two protocols: once
        ax.annotate(', '.join(names), (xs[right], ys[right]), fontsize=7.5,
                    xytext=(5, -3), textcoords='offset points', color=INK_MUTED)


def _diagonal(ax, lim, label='y = x'):
    ax.plot([0, lim], [0, lim], ls='--', color=INK_MUTED, lw=1, zorder=1, label=label)
    ax.set(xlim=(0, lim), ylim=(0, lim), aspect='equal')


def _by_difficulty(ax, data, x, y):
    for diff, grp in data.groupby('difficulty'):
        ax.scatter(grp[x], grp[y], s=70, color=DIFF_COLOR[diff], edgecolor='white', linewidth=1.5,
                   label=diff, zorder=3)
    label_points(ax, data[x], data[y], data.ref_code, tol=0.03)


def _irmsd_panel(ax, irm, values, title, lim=2.2):
    """Published vs computed I-RMSD; points past `lim` are drawn at the edge with their value."""
    for diff, grp in irm.groupby('difficulty'):
        inside = grp[values[grp.index] <= lim]
        ax.scatter(values[inside.index], inside.benchmark_irmsd, s=70, color=DIFF_COLOR[diff],
                   edgecolor='white', linewidth=1.5, label=diff, zorder=3)
        for idx, row in grp[values[grp.index] > lim].iterrows():
            ax.scatter(lim, row.benchmark_irmsd, s=90, marker='>', color=DIFF_COLOR[diff],
                       edgecolor='white', linewidth=1.5, zorder=3, clip_on=False)
            ax.annotate(f'{row.ref_code} {values[idx]:.2f} Å', (lim, row.benchmark_irmsd), fontsize=8,
                        xytext=(-8, 8), textcoords='offset points', ha='right', color=INK)
    inside = values <= lim
    label_points(ax, values[inside], irm.benchmark_irmsd[inside], irm.ref_code[inside])
    _diagonal(ax, lim)
    ax.set(title=title, xlabel='computed I-RMSD (Å)', ylabel='published I-RMSD (Å)')
    ax.legend(loc='lower right', fontsize=8)


def plot_irmsd_vs_published(irm):
    """Figure 1: the I-RMSD of the benchmark files and of the workflow against the published one."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    _irmsd_panel(axes[0], irm, irm.bench_irmsd, 'benchmark files: the definition, reproduced')
    _irmsd_panel(axes[1], irm, irm.wf_irmsd, 'workflow structures: what HADDOCK3 docks')
    fig.suptitle('I-RMSD (Cα, 10 Å interface, both partners) against Benchmark 5.5', y=0.99)
    _save(fig, 'fig1_irmsd_vs_published.png')


def plot_irmsd_sides(irm):
    """Figure 2: the antibody and antigen sides of the I-RMSD, benchmark files."""
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    _by_difficulty(ax, irm, 'bench_irmsd_ab', 'bench_irmsd_ag')
    _diagonal(ax, 1.1 * max(irm.bench_irmsd_ab.max(), irm.bench_irmsd_ag.max()))
    ax.set(xlabel='antibody side (Å)', ylabel='antigen side (Å)',
           title='The two sides of the I-RMSD (benchmark files)')
    ax.legend(loc='lower right', fontsize=8)
    _save(fig, 'fig2_irmsd_sides.png')


def plot_case_bars(summary):
    """Figure 3: per case, the antibody (framework, Fv, interface, CDRs) and the antigen (the ligand of
    the benchmark: whole and interface), with the published I-RMSD of both partners on each panel.

    The antigen's two bars take the colours of the antibody's whole Fv and interface, the same extents.
    The whole antigen of 3HMX, 3G6D, 3V6Z, 2VXT and the trimers reaches past the antibody scale, so each
    panel has its own.
    """
    antibody = [('framework', 'framework (self)'), ('fv', 'whole Fv (self)'),
                ('interface_ab', 'interface, antibody side (self)'), ('cdr_fw', 'all CDRs (framework frame)'),
                ('H3_fw', 'CDR-H3 (framework frame)')]
    antigen = [('wf_antigen', 'whole antigen (self)', SERIES[1]),
               ('wf_irmsd_ag', 'interface, antigen side (self)', SERIES[2])]
    x, width = np.arange(len(summary)), 0.16
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, height_ratios=(1.3, 1))
    for i, (key, label) in enumerate(antibody):
        top.bar(x + (i - 2) * width, summary[key], width, label=label, color=SERIES[i],
                edgecolor='white', linewidth=1)
    for i, (key, label, color) in enumerate(antigen):
        bottom.bar(x + (i - 0.5) * width, summary[key], width, label=label, color=color,
                   edgecolor='white', linewidth=1)
    for ax in (top, bottom):
        ax.plot(x, summary.benchmark_irmsd, 'D', color=INK, ms=7, label='published I-RMSD (both partners)', zorder=5)
        ax.set(ylabel='Cα RMSD (Å)')
        ax.legend(fontsize=9, ncol=2, loc='upper left')
    top.set_title('Bound vs unbound antibody: framework, Fv, interface and CDR loops')
    bottom.set_title('Bound vs unbound antigen (the ligand): whole and interface')
    bottom.set_xticks(x, [f'{r.ref_code}/{r.unb_code}\n{r.difficulty}' for _, r in summary.iterrows()], fontsize=8)
    bottom.set_xlabel('case (sorted by published I-RMSD)')
    _save(fig, 'fig3_per_case_bars.png')


def plot_cdr_heatmap(df, summary):
    """Figure 4: per-CDR RMSD per case, displacement and shape change."""
    labels = [f'{r.ref_code}/{r.unb_code}' for _, r in summary.iterrows()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    for ax, scheme, title in zip(axes, ('fv_fw', 'self'),
                                 ('displacement (framework frame)', 'internal shape change (self-fit)')):
        mat = pd.DataFrame({c: pick(df, c, scheme) for c in CDRS}).loc[summary.index]
        sns.heatmap(mat, annot=True, fmt='.2f', cmap=BLUES, vmin=0, vmax=4.1, ax=ax,
                    cbar_kws={'label': 'Cα RMSD (Å)'}, annot_kws={'fontsize': 8},
                    linewidths=1, linecolor='white')
        ax.set(title=f'CDR {title}', xlabel='CDR (IMGT)', ylabel='')
        ax.set_yticks(np.arange(len(labels)) + 0.5, labels, rotation=0, fontsize=8)
    axes[0].set_ylabel('case (sorted by published I-RMSD)')
    fig.suptitle('Per-CDR bound vs unbound Cα RMSD. Blank: CDR-L2 is 3 residues wide, too few for a self-fit',
                 y=1.0, fontsize=10)
    _save(fig, 'fig4_cdr_heatmap.png')


def plot_difficulty(summary):
    """Figure 5: antibody RMSDs split by benchmark difficulty; returns the medians."""
    panels = [('wf_irmsd_ab', 'interface, antibody side\n(self)'), ('fv', 'whole Fv\n(self)'),
              ('cdr_fw', 'all CDRs\n(framework frame)'), ('H3_fw', 'CDR-H3\n(framework frame)')]
    order = ['rigid', 'medium']
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 5), sharey=True)
    for ax, (key, label) in zip(axes, panels):
        sns.boxplot(data=summary, x='difficulty', y=key, order=order, hue='difficulty', legend=False, ax=ax,
                    palette=DIFF_COLOR, width=0.55, showfliers=False, boxprops={'alpha': 0.45})
        sns.stripplot(data=summary, x='difficulty', y=key, order=order, hue='difficulty', legend=False, ax=ax,
                      palette=DIFF_COLOR, size=8, edgecolor='white', linewidth=1, jitter=0.18)
        u = stats.mannwhitneyu(summary.loc[summary.difficulty == 'rigid', key],
                               summary.loc[summary.difficulty == 'medium', key], alternative='two-sided')
        ax.set(title=f'{label}\nMann-Whitney p = {u.pvalue:.3f}', xlabel='', ylabel='')
    axes[0].set_ylabel('Cα RMSD (Å)')
    n = summary.difficulty.value_counts()
    fig.suptitle(f"Benchmark difficulty class vs antibody conformational change "
                 f"({n.get('rigid', 0)} rigid, {n.get('medium', 0)} medium)", y=1.02)
    _save(fig, 'fig5_difficulty_split.png')
    return summary.groupby('difficulty')[[k for k, _ in panels]].median().round(2)


def plot_elbow(summary):
    """Figure 6: the Fv in a whole-Fab fit, and the CDR motion split into displacement and shape."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6))
    for ax, x, y, line in ((ax1, 'fv', 'fv_in_full', 'y = x (no elbow change)'),
                           (ax2, 'cdr_self', 'cdr_fw', 'y = x (pure shape change)')):
        _by_difficulty(ax, summary, x, y)
        _diagonal(ax, 1.1 * max(summary[x].max(), summary[y].max()), line)
        ax.legend(loc='lower right', fontsize=9)
    ax1.set(xlabel='Fv Cα RMSD, fitted on the Fv (Å)', ylabel='Fv Cα RMSD, fitted on the whole Fab (Å)',
            title='Fitting the whole Fab misplaces the Fv:\nthe elbow angle differs between the entries')
    ax2.set(xlabel='CDR Cα RMSD, fitted on the CDRs (Å)', ylabel='CDR Cα RMSD in the framework frame (Å)',
            title='CDR motion decomposed:\nabove the line = rigid displacement of the loops')
    _save(fig, 'fig6_elbow_and_decomposition.png')


def plot_h3_profile(w):
    """Figure 7: the displacement of every CDR-H3 residue of the worst case, interface marked."""
    h3, case = w['h3'], w['case']
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(h3.imgt, h3.dev_ca_fv_fw, color=[SERIES[1] if i else SERIES[0] for i in h3.in_interface],
           edgecolor='white', linewidth=1)
    for i, (dev, name) in enumerate(zip(h3.dev_ca_fv_fw, h3.resname)):
        ax.annotate(seq1(name), (i, dev), ha='center', va='bottom', fontsize=9, color=INK)
    ax.axhline(w['rmsd'], ls='--', color=INK_MUTED, lw=1, label=f"loop RMSD = {w['rmsd']:.2f} Å")
    handles = [Patch(color=SERIES[1], label='in the 10 Å interface'), Patch(color=SERIES[0], label='outside it')]
    ax.set(xlabel='IMGT position', ylabel='Cα displacement (Å)',
           title=f'CDR-H3 residue by residue, {case.ref_code} vs {case.unb_code} (framework frame)')
    ax.legend(handles=ax.get_legend_handles_labels()[0] + handles, fontsize=9)
    _save(fig, 'fig7_worst_h3_profile.png')


def plot_docking_gain(gain, capped=()):
    """Figure 8: what each CDR ensemble changes in the docking against the CDR displacement.

    Both ensembles of a case sit on one vertical (the case's displacement), joined by a thin line.
    `capped` are the case indices still scored against the file-order reference
    (FILE_ORDER_REFERENCE), drawn hollow.
    """
    gain = gain.dropna(subset=['d_acceptable_awh'])
    hollow = gain.index.isin(capped)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, metric, scale, ylabel in ((axes[0], 'acceptable', 100, 'change in acceptable models (% points)'),
                                      (axes[1], 'best_dockq', 1, 'change in best DockQ')):
        ys = {p: scale * gain[f'd_{metric}_{p}'] for p in ('md', 'awh')}
        ax.vlines(gain.cdr_fw, ys['md'], ys['awh'], color=GRID, lw=1.5, zorder=1)
        for p, y in ys.items():
            ax.scatter(gain.cdr_fw[~hollow], y[~hollow], s=60, color=PROTOCOL_COLOR[p], edgecolor='white',
                       linewidth=1.2, zorder=3, label=f'{PROTOCOL_LABEL[p]} − baseline')
            ax.scatter(gain.cdr_fw[hollow], y[hollow], s=60, facecolor='white', edgecolor=PROTOCOL_COLOR[p],
                       linewidth=1.5, zorder=3)
        # the hollow points are named in the legend
        top = np.maximum(ys['md'], ys['awh'])
        label_points(ax, gain.cdr_fw[~hollow], top[~hollow], gain.ref_code[~hollow], tol=0.07, same_units=False)
        ax.axhline(0, color=INK_MUTED, lw=1, zorder=1)
        ax.set(xlabel='CDR Cα displacement, framework frame (Å)', ylabel=ylabel)
    if hollow.any():
        axes[0].scatter([], [], s=60, facecolor='white', edgecolor=INK_MUTED, linewidth=1.5,
                        label=f"reference caps DockQ: {', '.join(gain.ref_code[hollow])}")
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    axes[0].set_title(f'Sampling: models with DockQ ≥ {ACCEPTABLE}, of 200')
    axes[1].set_title('Quality: the best model')
    fig.suptitle('Does a larger bound-vs-unbound CDR change make the CDR ensembles pay off?', y=1.0)
    _save(fig, 'fig8_docking_gain_vs_cdr.png')


def plot_lrmsd_vs_elbow(le):
    """Figure 9: the L-RMSD of the right-interface models against the elbow of the docked structures."""
    le = le.dropna(subset=['lrmsd_right'])
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for protocol in DOCKING_RUNS:
        grp = le[le.protocol == protocol]
        ax.scatter(grp.elbow, grp.lrmsd_right, s=70, color=PROTOCOL_COLOR[protocol], edgecolor='white',
                   linewidth=1.5, zorder=3, label=PROTOCOL_LABEL[protocol])
    label_points(ax, le.elbow, le.lrmsd_right, le.ref_code, tol=0.05, same_units=False)
    r = stats.spearmanr(le.elbow, le.lrmsd_right)
    ax.set(xlabel='Fv Cα RMSD to bound in a whole-Fab fit, docked structures (median, Å)',
           ylabel=f'L-RMSD of the models with i-RMSD ≤ {RIGHT_INTERFACE:g} Å (median, Å)',
           title=f'With the interface right, the L-RMSD measures the elbow\n'
                 f'Spearman ρ = {r.statistic:+.2f} over {len(le)} case × protocol points')
    ax.set_xlim(0, 1.1 * le.elbow.max())
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', fontsize=9)
    _save(fig, 'fig9_lrmsd_vs_elbow.png')


# The per-structure measures figure 10 and report_ensembles show, with their axis labels
ENSEMBLE_MEASURES = {'cdr_fw': 'CDR Cα RMSD to bound,\nframework frame (Å)',
                     'paratope_sc': 'paratope side chains\nRMSD to bound (Å)',
                     'fv_in_full': 'Fv Cα RMSD to bound,\nwhole-Fab fit (Å)',
                     'paratope_chi1_off': 'paratope χ1 more than\n40° off (fraction)'}


def plot_ensembles(ens, gain, capped=()):
    """Figure 10: what each ensemble gains, and how far the structures it docked are from the bound antibody.

    Cases are ordered by the AWH gain in acceptable models; the `capped` ones are left out. Below the
    gains, every docked structure: the unbound antibody (a bar) and the 20 members of each ensemble
    (dots), by the CDR displacement in the framework frame and by the paratope side chains, each on
    its own backbone.
    """
    gain = gain.dropna(subset=['d_acceptable_awh']).drop(capped, errors='ignore').sort_values('d_acceptable_awh')
    x = np.arange(len(gain))
    rows = ('cdr_fw', 'paratope_sc')
    fig, axes = plt.subplots(1 + len(rows), 1, figsize=(13, 11), sharex=True, height_ratios=(1, 1.4, 1.4))
    unbound = ens[ens.source == 'unbound'].set_index('case_idx')
    for shift, protocol in ((-0.18, 'md'), (0.18, 'awh')):
        axes[0].bar(x + shift, 100 * gain[f'd_acceptable_{protocol}'], width=0.34, color=PROTOCOL_COLOR[protocol],
                    edgecolor='white', linewidth=1, label=f'{PROTOCOL_LABEL[protocol]} − baseline')
        members = ens[ens.source == protocol]
        for ax, name in zip(axes[1:], rows):
            for i, idx in enumerate(gain.index):
                y = members[name][members.case_idx == idx]
                ax.scatter(np.full(len(y), i + shift) + np.linspace(-0.07, 0.07, len(y)), y, s=14,
                           color=PROTOCOL_COLOR[protocol], alpha=0.8, linewidth=0, zorder=3,
                           label=PROTOCOL_LABEL[protocol] if i == 0 else None)
    for ax, name in zip(axes[1:], rows):
        ax.hlines(unbound[name][gain.index], x - 0.38, x + 0.38, color=PROTOCOL_COLOR['baseline'], lw=2.5,
                  zorder=4, label='unbound antibody (baseline docking)')
        # headroom for the legend above the highest structure
        ax.set(ylabel=ENSEMBLE_MEASURES[name], ylim=(0, 1.2 * ens.loc[ens.case_idx.isin(gain.index), name].max()))
        ax.legend(loc='upper left', fontsize=9, ncol=3, frameon=False)
    axes[0].axhline(0, color=INK_MUTED, lw=1)
    axes[0].set(ylabel='change in acceptable\nmodels (% points)',
                title='What each ensemble gains over the baseline docking, and what it docked')
    axes[0].legend(loc='upper left', fontsize=9, frameon=False)
    axes[-1].set_xticks(x, gain.ref_code)
    _save(fig, 'fig10_ensembles_vs_gain.png')


def report_docking_gain(gain, capped=()):
    """Spearman correlation of the CDR displacement with what each ensemble changes.

    The cases in `capped` are left out: their DockQ is not comparable.
    """
    if len(capped):
        print('left out, their reference caps DockQ:', ', '.join(gain.ref_code[gain.index.isin(capped)]))
    for protocol in ('md', 'awh'):
        g = gain.dropna(subset=[f'd_acceptable_{protocol}']).drop(capped, errors='ignore')
        parts = []
        for metric, label in (('acceptable', 'acceptable models'), ('best_dockq', 'best DockQ'),
                              ('top', f'acceptable in top {TOP}')):
            r = stats.spearmanr(g.cdr_fw, g[f'd_{metric}_{protocol}'])
            parts.append(f'{label} rho = {r.statistic:+.2f} (p = {r.pvalue:.2f})')
        print(f'{PROTOCOL_LABEL[protocol]} vs CDR displacement, {len(g)} cases: ' + ', '.join(parts))
        d = g[f'd_acceptable_{protocol}']
        print(f'    more acceptable models than the baseline in {int((d > 0.005).sum())}, fewer in '
              f'{int((d < -0.005).sum())}, as many in {int((d.abs() <= 0.005).sum())}')


def report_dockq_terms(terms, capped=()):
    """Which DockQ terms carry the change of each ensemble, summed over the comparable cases."""
    t = terms.drop(capped, errors='ignore')
    for protocol in ('md', 'awh'):
        parts = ', '.join(f'{name} {t[f"d_{name}_{protocol}"].sum() / 3:+.2f}' for name in ('fnat', 'irmsd', 'lrmsd'))
        print(f'{PROTOCOL_LABEL[protocol]}, {len(t)} cases: mean DockQ {t[f"d_dockq_{protocol}"].sum():+.2f} '
              f'in all = {parts} (a third of each term)')
    big = t.d_dockq_awh.nlargest(2)
    print(f'largest AWH changes: ' + ', '.join(f'{t.ref_code[i]} {v:+.2f}' for i, v in big.items())
          + f' of {t.d_dockq_awh.clip(lower=0).sum():+.2f} gained in all')


def report_lrmsd_vs_elbow(le):
    """The L-RMSD of the right-interface models against the elbow, pooled and per protocol."""
    v = le.dropna(subset=['lrmsd_right'])
    r = stats.spearmanr(v.elbow, v.lrmsd_right)
    print(f'pooled: rho = {r.statistic:+.2f} (p = {r.pvalue:.1g}), {len(v)} case x protocol points '
          f'with at least {MIN_RIGHT} models of i-RMSD <= {RIGHT_INTERFACE:g} A')
    for protocol in DOCKING_RUNS:
        grp = v[v.protocol == protocol]
        r = stats.spearmanr(grp.elbow, grp.lrmsd_right)
        print(f'    {PROTOCOL_LABEL[protocol]:22s}: rho = {r.statistic:+.2f} (p = {r.pvalue:.2f}), {len(grp)} cases')


def report_ensembles(gain, et, capped=()):
    """What the AWH gain does and does not follow; returns one row of tests per ENSEMBLE_MEASURES.

    Per measure (lower is closer to bound): in how many cases the AWH ensemble's median member is
    closer than the unbound antibody and than the free-MD median, in how many its closest member is
    closer than the unbound one, and the Spearman correlation of the AWH gain in acceptable models with
    the AWH - unbound change (median member, closest member), and of the AWH - MD difference in
    acceptable models with the AWH - MD difference of the medians. Then the pose the AWH starts from.
    """
    g = gain.dropna(subset=['d_acceptable_awh']).drop(capped, errors='ignore')
    e = et.loc[g.index]
    d_acc = g.acceptable_awh - g.acceptable_md
    n = len(g)
    print(f'{n} cases with both ensembles, capped ones left out; AWH more acceptable models than free MD in '
          f'{int((d_acc > 0.005).sum())}/{n}, fewer in {int((d_acc < -0.005).sum())}')
    rows = []
    for name in ENSEMBLE_MEASURES:
        tests = {'AWH gain ~ AWH − unbound (median)': (g.d_acceptable_awh, e[f'{name}_awh'] - e[f'{name}_unbound']),
                 'AWH gain ~ AWH − unbound (closest)': (g.d_acceptable_awh, e[f'{name}_awh_min'] - e[f'{name}_unbound']),
                 'AWH − MD ~ AWH − MD (median)': (d_acc, e[f'{name}_awh'] - e[f'{name}_md'])}
        row = {'measure': name, 'cases': n,
               'AWH median closer than unbound': int((e[f'{name}_awh'] < e[f'{name}_unbound']).sum()),
               'AWH median closer than MD': int((e[f'{name}_awh'] < e[f'{name}_md']).sum()),
               'AWH closest closer than unbound': int((e[f'{name}_awh_min'] < e[f'{name}_unbound']).sum())}
        for label, (y, x) in tests.items():
            r = stats.spearmanr(y, x)
            row[f'ρ {label}'], row[f'p {label}'] = r.statistic, r.pvalue
        rows.append(row)

    def rho(x, y, label):
        r = stats.spearmanr(x, y)
        print(f'    {label}: rho = {r.statistic:+.2f} (p = {r.pvalue:.2f})')
    print('The pose stage 3 starts the AWH from (the baseline cluster_1_model_1):')
    rho(g.awh_start_dockq, g.d_acceptable_awh, 'its DockQ vs the AWH gain                  ')
    rho(g.awh_start_dockq, g.acceptable_baseline, 'its DockQ vs the baseline acceptable models')
    rho(g.awh_start_dockq, g.acceptable_awh, 'its DockQ vs the AWH acceptable models     ')
    wrong = g[g.awh_start_dockq < ACCEPTABLE]
    print(f'    started from an incorrect pose (DockQ < {ACCEPTABLE}): ' + ', '.join(
        f'{r.ref_code} {r.awh_start_dockq:.2f} (acceptable {100 * r.acceptable_baseline:.0f}% -> {100 * r.acceptable_awh:.0f}%)'
        for _, r in wrong.iterrows()))
    return pd.DataFrame(rows).set_index('measure')
