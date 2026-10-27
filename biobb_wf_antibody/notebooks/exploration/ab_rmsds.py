"""Helpers of ab_rmsds.ipynb: how far the unbound antibody of each benchmark case is from its bound form.

The structures are the workflow's own. Every case is prepared by stage 1 from the configuration
array/launch_wf.py writes for it, so what is measured is what HADDOCK3 docks: the assembly or the
model stage 0 downloads, the unbound antibody and antigen repaired by Modeller, and the reference
complex fused into chain A (antibody) and chain B (antigen). The residue correspondence is
utils.map_contact_residues, the interface is haddock_interface and the CDRs come from ANARCII
through cdr.imgt_region_ri, the calls stages 1 and 2 make.

What is left here is what only the notebook needs: the published benchmark values, the I-RMSD as
the benchmark defines it, the RMSD of the antibody regions and the per-case tables.
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

import MDAnalysis as mda
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
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
    property below is the run's own.
    """
    if idx not in _CONFS:
        case_dir = PREP / f'case_{idx}'
        config = case_dir / 'workflow.yml'
        if not config.is_file():
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

    Done once: the Modeller repair takes minutes, so existing outputs are kept unless `force`.
    Returns case_files(idx).
    """
    c = case_config(idx)
    files = case_files(idx)
    if _modeller_key():
        os.environ.setdefault('MODELLER_KEY', _modeller_key())
    # biobb writes its log files and sandboxes to the working directory
    with fu.change_dir(str(c['dir'])):
        prepared = ('antibody', 'antigen', 'reference')
        if force or not all(Path(files[k]).is_file() for k in prepared):
            download_pdbs(c['log'], c['prop'], c['paths'], c['ids'])
            prepare_structures(c['log'], c['prop'], c['paths'], c['ids'])
        if force or not Path(files['interface']).is_file():
            interface_residues(files['reference'], files['interface'])
        anarcii_chain_types(c['paths']['step2_19_anarcii'], c['prop']['step2_19_anarcii'], force)
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
# Figures
# ============================================================================

# Categorical slots 1-2 and 1-5 of the reference data-viz palette, validated for colour-vision
# deficiency all-pairs (2 classes) and adjacent-pairs (5 bars); the 5 bars carry a legend and the
# summary table is the table view
DIFF_COLOR = {'rigid': '#2a78d6', 'medium': '#eb6834'}
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK_MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
# Sequential blue ramp, light -> dark, for magnitudes
BLUES = LinearSegmentedColormap.from_list(
    'blues', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])


def label_points(ax, xs, ys, labels, tol=0.03):
    """Label the points of a scatter; points closer than `tol` of the data span share one label."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    span = max(np.ptp(xs), np.ptp(ys)) or 1.0
    groups = []
    for i in np.argsort(xs):
        for group in groups:
            if np.hypot(xs[i] - xs[group[0]], ys[i] - ys[group[0]]) < tol * span:
                group.append(i)
                break
        else:
            groups.append([i])
    for group in groups:
        right = max(group, key=lambda i: xs[i])
        ax.annotate(', '.join(str(list(labels)[i]) for i in group), (xs[right], ys[right]), fontsize=7.5,
                    xytext=(5, -3), textcoords='offset points', color=INK_MUTED)


# ============================================================================
# Tables
# ============================================================================

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
