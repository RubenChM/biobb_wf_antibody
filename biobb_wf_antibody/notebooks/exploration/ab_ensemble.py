"""Choosing which MD conformers to hand to HADDOCK3.

The workflow currently selects them with `gmx cluster` on the CDR C-alpha atoms. That is a
*geometric* criterion: an RMSD cutoff splits broad basins and merges conformers separated by a
real barrier, it cannot see side chains at all, and its cutoff is a free parameter. This module
implements the alternative -- TICA on torsion features, discretisation, an MSM, and PCCA+
metastable states as the conformers -- plus the scoring machinery needed to decide whether it is
actually better.

Everything here is per case and per trajectory. The angle definitions come from `ab_common`, the
same ones `ab_torsions.ipynb` used on the crystal structures, because the coverage metric
compares an MD frame with a bound crystal structure and two different chi1 atom sets would make
that comparison quietly meaningless.

The scoring idea: the bound structure is known for all 16 benchmark cases, so a selection method
does not have to be argued about. For an ensemble of N members,

    d_min(N) = min over the N members of (CDR C-alpha RMSD to the bound antibody,
               measured after superposing on the Fv framework C-alpha)

is exactly the `('cdr_all', 'ca', 'fv_fw_ca')` quantity of `ab_rmsds.ipynb`, evaluated against a
trajectory frame instead of the unbound crystal structure. Lower is better, and the curve against
N is directly comparable between selection strategies.
"""

from __future__ import annotations

import warnings

import MDAnalysis as mda
import numpy as np
import pandas as pd
from Bio.SVDSuperimposer import SVDSuperimposer
from MDAnalysis.analysis.dihedrals import Dihedral

import biobb_wf_antibody.notebooks.exploration.ab_common as ab
from biobb_wf_antibody.notebooks.exploration.ab_common import PEPTIDE_BOND_MAX, chi_quads, imgt_region, res_key

warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis')

# Below this many frames the kinetic quantities (timescales, populations, PCCA+ states) are
# noise. The pipeline still runs, so the code paths can be exercised on a test trajectory, but
# everything that depends on statistics is flagged rather than reported.
MIN_FRAMES_MEANINGFUL = 500


# --- joining the MD system to the benchmark case --------------------------------------------

def md_residues(universe) -> list:
    """Protein residues of an MD universe, in file order."""
    return list(universe.select_atoms('protein').residues)


def md_prep_mapping(universe, prep_unb_pdb, chain_order=None) -> dict:
    """{res_key of a prep residue: 0-based MD residue index}, validated by sequence.

    The MD system is built from the same cleaned unbound antibody the RMSD/torsion notebooks
    prepared, so the residues correspond one to one -- but the MD file has lost the chain IDs
    (pdb2gmx merges everything into one segment) and the chain order is not guaranteed. Both are
    recovered by matching the concatenated sequences, and a mismatch raises instead of silently
    shifting every residue by one.
    """
    obs = ab.observed_residues(prep_unb_pdb)
    residues = md_residues(universe)
    md_seq = ''.join(ab.AA3TO1.get(r.resname.upper(), 'X') for r in residues)

    import itertools
    orders = [chain_order] if chain_order else itertools.permutations(obs)
    for order in orders:
        if ''.join(obs[c][0] for c in order) == md_seq:
            mapping, i = {}, 0
            for c in order:
                for residue in obs[c][1]:
                    mapping[res_key(residue)] = i
                    i += 1
            return {'map': mapping, 'chain_order': tuple(order), 'n_res': len(residues)}
    raise ValueError(f'no chain order of {list(obs)} reproduces the MD sequence '
                     f'({len(md_seq)} residues); the MD system is not this prepared antibody')


# --- features -------------------------------------------------------------------------------

def feature_spec(ctx: dict, universe, mapping: dict, iface: str = 'interface10',
                 use_chi1: bool = True) -> pd.DataFrame:
    """One row per torsion to featurise: which MD atoms, and what it is.

    CDR `phi`/`psi` for every matched CDR position, plus `chi1` of every interface residue that
    has one. `phi` and `psi` need the neighbouring residue, and the MD system is a single segment
    in which the light and heavy chains are adjacent, so the peptide bond is checked
    geometrically -- exactly as in `ab_torsions.ipynb` -- rather than assumed from adjacency.
    """
    residues = md_residues(universe)
    rows = []
    for slot, key, ref_res, unb_res in ctx['imgt_pairs']:
        idx = mapping['map'].get(res_key(unb_res))
        if idx is None:
            continue
        region = imgt_region(key[0])
        in_cdr = region.startswith('CDR')
        in_iface = res_key(ref_res) in ctx[iface]
        if not (in_cdr or (use_chi1 and in_iface)):
            continue
        res = residues[idx]
        chain_type = 'H' if ctx['slot_is_heavy'][slot] else 'L'
        label = f'{chain_type}{key[0]}{key[1]}'
        want = []
        if in_cdr:
            want += [('phi', None), ('psi', None)]
        if use_chi1 and in_iface:
            quad = chi_quads(res.resname).get('chi1')
            if quad:
                want.append(('chi1', quad))
        for angle, quad in want:
            atoms = _quad_atoms(residues, idx, angle, quad)
            if atoms is None:
                continue
            rows.append({'angle': angle, 'md_index': idx, 'imgt_num': key[0],
                         'imgt_icode': key[1], 'chain_type': chain_type, 'region': region,
                         'in_cdr': in_cdr, 'in_iface': in_iface, 'resname': res.resname,
                         'label': f'{label}.{angle}', 'atoms': atoms})
    return pd.DataFrame(rows)


def _bonded(res_a, res_b) -> bool:
    """C(a)-N(b) below the peptide-bond cutoff, on the current frame."""
    try:
        c = res_a.atoms.select_atoms('name C')[0].position
        n = res_b.atoms.select_atoms('name N')[0].position
    except IndexError:
        return False
    return float(np.linalg.norm(c - n)) < PEPTIDE_BOND_MAX


def _quad_atoms(residues, idx, angle, quad):
    """The four MDAnalysis Atoms of one torsion, or None if it is not defined."""
    res = residues[idx]

    def pick(r, name):
        sel = r.atoms.select_atoms(f'name {name}')
        return sel[0] if len(sel) else None

    if angle == 'phi':
        if idx == 0 or not _bonded(residues[idx - 1], res):
            return None
        spec = [(residues[idx - 1], 'C'), (res, 'N'), (res, 'CA'), (res, 'C')]
    elif angle == 'psi':
        if idx + 1 >= len(residues) or not _bonded(res, residues[idx + 1]):
            return None
        spec = [(res, 'N'), (res, 'CA'), (res, 'C'), (residues[idx + 1], 'N')]
    else:
        spec = [(res, n) for n in quad]
    atoms = [pick(r, n) for r, n in spec]
    return None if any(a is None for a in atoms) else atoms


def featurize(universe, spec: pd.DataFrame, verbose: bool = True):
    """(n_frames, n_torsions) array of angles in degrees, in the order of `spec`."""
    groups = [mda.AtomGroup(row.atoms) for row in spec.itertuples()]
    result = Dihedral(groups).run(verbose=False)
    angles = np.asarray(result.results.angles, dtype=float)
    if verbose:
        n_bb = int(((spec.angle != 'chi1') & spec.in_cdr).sum())
        print(f'featurised {angles.shape[0]} frames x {angles.shape[1]} torsions '
              f'({n_bb} CDR backbone, {int((spec.angle == "chi1").sum())} interface chi1)')
    return angles


def structure_angles(pdb_path) -> dict:
    """{res_key: {'phi'|'psi'|'chi1': degrees}} for a prepared crystal structure.

    Needed to place the bound (or unbound) crystal structure in the same feature space as the
    trajectory, which is how the "does the sampled ensemble reach the bound conformation at all"
    question gets answered. Only the three angles the trajectory features use.
    """
    out = {}
    for _cid, (_seq, residues) in ab.observed_residues(pdb_path).items():
        for i, res in enumerate(residues):
            angles = {}
            prev_res = residues[i - 1] if i > 0 else None
            next_res = residues[i + 1] if i + 1 < len(residues) else None
            spec = {}
            if prev_res is not None and _bonded_bio(prev_res, res):
                spec['phi'] = [(prev_res, 'C'), (res, 'N'), (res, 'CA'), (res, 'C')]
            if next_res is not None and _bonded_bio(res, next_res):
                spec['psi'] = [(res, 'N'), (res, 'CA'), (res, 'C'), (next_res, 'N')]
            quad = chi_quads(res.get_resname()).get('chi1')
            if quad:
                spec['chi1'] = [(res, n) for n in quad]
            for angle, quads in spec.items():
                if any(n not in r for r, n in quads):
                    continue
                angles[angle] = ab.dihedral(*[r[n].coord for r, n in quads])
            out[res_key(res)] = angles
    return out


def _bonded_bio(res_a, res_b) -> bool:
    if 'C' not in res_a or 'N' not in res_b:
        return False
    return float(np.linalg.norm(res_a['C'].coord - res_b['N'].coord)) < PEPTIDE_BOND_MAX


def featurize_structure(ctx: dict, spec: pd.DataFrame, which: str = 'ref'):
    """The same torsions as `spec`, evaluated on one crystal structure of the case.

    Returns (angles in degrees with NaN where undefined, n_missing). `which` is 'ref' for the
    bound antibody or 'unb' for the unbound one.
    """
    angles_by_res = structure_angles(ctx['paths'][f'{which}_ab'])
    lookup = {}
    for slot, key, ref_res, unb_res in ctx['imgt_pairs']:
        chain_type = 'H' if ctx['slot_is_heavy'][slot] else 'L'
        lookup[(chain_type, key[0], key[1])] = ref_res if which == 'ref' else unb_res
    out, missing = [], 0
    for row in spec.itertuples():
        res = lookup.get((row.chain_type, row.imgt_num, row.imgt_icode))
        value = np.nan
        if res is not None:
            value = angles_by_res.get(res_key(res), {}).get(row.angle, np.nan)
        missing += int(np.isnan(value))
        out.append(value)
    return np.asarray(out, dtype=float)[None, :], missing


def sincos(angles_deg) -> np.ndarray:
    """Periodicity-safe features: [sin(a), cos(a)] side by side.

    Raw angles must never be fed to a covariance-based method: the -179 -> +179 deg branch cut
    would enter the covariance as a 358 deg excursion.
    """
    rad = np.radians(np.asarray(angles_deg, dtype=float))
    return np.concatenate([np.sin(rad), np.cos(rad)], axis=1)


# --- TICA, discretisation, MSM ---------------------------------------------------------------

def tica(X, lag: int, var_cutoff: float = 0.95, dim: int | None = None):
    """Fit TICA on sin/cos features. Returns (model, projection, info)."""
    from deeptime.decomposition import TICA

    est = TICA(lagtime=lag, var_cutoff=None if dim else var_cutoff, dim=dim)
    model = est.fit(X).fetch_model()
    Y = model.transform(X)
    info = {'lag': lag, 'n_features': X.shape[1], 'n_tics': Y.shape[1],
            'timescales': np.asarray(model.timescales(lagtime=lag))[:Y.shape[1]],
            'singular_values': np.asarray(model.singular_values)[:Y.shape[1]]}
    return model, Y, info


def implied_timescales(X, lags, n_ts: int = 4, dim: int = 4) -> pd.DataFrame:
    """TICA timescales as a function of lag time -- the check that a lag choice is not arbitrary."""
    rows = []
    for lag in lags:
        if lag >= len(X):
            continue
        try:
            _m, _Y, info = tica(X, lag=lag, dim=min(dim, X.shape[1]))
        except Exception as exc:                     # rank-deficient at short data lengths
            rows.append({'lag': lag, 'error': f'{type(exc).__name__}'})
            continue
        row = {'lag': lag}
        for i, t in enumerate(info['timescales'][:n_ts]):
            row[f'its{i + 1}'] = float(t)
        rows.append(row)
    return pd.DataFrame(rows)


def discretise(Y, n_micro: int, seed: int = 0):
    """k-means microstates in TIC space. Returns (labels, cluster centres)."""
    from deeptime.clustering import KMeans

    n_micro = max(2, min(n_micro, len(Y) // 2))
    model = KMeans(n_clusters=n_micro, fixed_seed=seed, n_jobs=1).fit(Y).fetch_model()
    return np.asarray(model.transform(Y)).ravel(), np.asarray(model.cluster_centers), n_micro


def macrostates(labels, lag: int, n_macro: int):
    """MSM over the microstates, then PCCA+ into `n_macro` metastable sets.

    Returns (per-frame macrostate assignment, per-macrostate population, msm) or None if the
    count matrix is too sparse to give a connected MSM -- which is the normal outcome on a short
    test trajectory, and the caller is expected to fall back rather than pretend otherwise.
    """
    from deeptime.markov import TransitionCountEstimator
    from deeptime.markov.msm import MaximumLikelihoodMSM

    try:
        counts = TransitionCountEstimator(lagtime=lag, count_mode='sliding').fit(
            labels).fetch_model().submodel_largest()
        msm = MaximumLikelihoodMSM(reversible=True).fit(counts).fetch_model()
        n_macro = max(2, min(n_macro, msm.n_states - 1))
        pcca = msm.pcca(n_macro)
        # msm states are a subset of the microstates; map back to every frame
        micro_to_macro = {state: int(m) for state, m in
                          zip(counts.state_symbols, pcca.assignments)}
        assign = np.array([micro_to_macro.get(int(s), -1) for s in labels])
        # stationary weight of a macrostate = sum over the microstates assigned to it
        pop = np.array([msm.stationary_distribution[
            [i for i, s in enumerate(counts.state_symbols) if micro_to_macro[s] == m]].sum()
            for m in range(n_macro)])
        return {'assign': assign, 'populations': pop, 'msm': msm, 'n_macro': n_macro,
                'timescales': np.asarray(msm.timescales())}
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {exc}'}


# --- selection strategies -------------------------------------------------------------------

def select_pcca(Y, macro: dict, n: int) -> list[int]:
    """One frame per metastable state -- the frame nearest that state's centre in TIC space.

    States are taken in order of decreasing stationary population, so truncating the list at N
    keeps the N most populated conformers.
    """
    assign, pop = macro['assign'], macro['populations']
    order = np.argsort(-pop)
    out = []
    for m in order:
        members = np.flatnonzero(assign == m)
        if not len(members):
            continue
        centre = Y[members].mean(axis=0)
        out.append(int(members[np.argmin(np.linalg.norm(Y[members] - centre, axis=1))]))
        if len(out) >= n:
            break
    return out


def select_kmedoid(Y, n: int, seed: int = 0) -> list[int]:
    """k-means in TIC space, then the frame closest to each centre. No kinetics involved."""
    from deeptime.clustering import KMeans

    n = max(1, min(n, len(Y)))
    model = KMeans(n_clusters=n, fixed_seed=seed, n_jobs=1).fit(Y).fetch_model()
    return [int(np.argmin(np.linalg.norm(Y - c, axis=1))) for c in model.cluster_centers]


def select_random(n_frames: int, n: int, rng) -> list[int]:
    return sorted(rng.choice(n_frames, size=min(n, n_frames), replace=False).tolist())


def select_stride(n_frames: int, n: int) -> list[int]:
    return np.unique(np.linspace(0, n_frames - 1, min(n, n_frames)).astype(int)).tolist()


# --- scoring against the bound structure -----------------------------------------------------

def bound_reference(ctx: dict, mapping: dict) -> dict:
    """Bound-structure C-alpha coordinates and the matching MD residue indices.

    Two sets: the Fv framework (what gets superposed) and the CDRs (what gets measured). Only
    positions whose CA exists in both structures are kept.
    """
    fw_bound, fw_md, cdr_bound, cdr_md = [], [], [], []
    for slot, key, ref_res, unb_res in ctx['imgt_pairs']:
        idx = mapping['map'].get(res_key(unb_res))
        if idx is None or 'CA' not in ref_res:
            continue
        region = imgt_region(key[0])
        if region == 'FR':
            fw_bound.append(ref_res['CA'].coord)
            fw_md.append(idx)
        elif region.startswith('CDR'):
            cdr_bound.append(ref_res['CA'].coord)
            cdr_md.append(idx)
    return {'fw_bound': np.asarray(fw_bound, float), 'fw_md': np.asarray(fw_md, int),
            'cdr_bound': np.asarray(cdr_bound, float), 'cdr_md': np.asarray(cdr_md, int)}


def cdr_rmsd_to_bound(universe, reference: dict, frames=None) -> np.ndarray:
    """CDR C-alpha RMSD to the bound antibody, per frame, in the Fv-framework frame.

    Superposes each frame's framework CA onto the bound framework CA, applies that transform to
    the CDR CA without refitting, and measures. This is `ab_rmsds.ipynb`'s `fv_fw_ca` scheme, so
    the numbers are on the same scale as its `cdr_ca_fw_rmsd` column.
    """
    ca = universe.select_atoms('protein and name CA')
    if len(ca) < int(reference['fw_md'].max()) + 1:
        raise RuntimeError(f'{len(ca)} CA atoms but the reference indexes up to '
                           f'{int(reference["fw_md"].max())}')
    frames = range(len(universe.trajectory)) if frames is None else frames
    sup = SVDSuperimposer()
    out = []
    for f in frames:
        universe.trajectory[int(f)]
        pos = ca.positions
        sup.set(reference['fw_bound'], pos[reference['fw_md']])
        sup.run()
        rot, tran = sup.get_rotran()
        moved = np.dot(pos[reference['cdr_md']], rot) + tran
        out.append(float(np.sqrt(np.mean(np.sum((reference['cdr_bound'] - moved) ** 2, axis=1)))))
    return np.asarray(out)


def score_pdb_models(pdb_path, reference: dict) -> np.ndarray:
    """Same metric, applied to the models of a multi-model PDB.

    Used for the incumbent: `gmx cluster` writes its representatives as MODEL records, so they
    can be scored directly, without having to work out which trajectory frame each one was.
    """
    u = mda.Universe(str(pdb_path))
    return cdr_rmsd_to_bound(u, reference)


def n_models(pdb_path) -> int:
    with open(pdb_path) as fh:
        return sum(1 for line in fh if line.startswith('MODEL')) or 1


# --- synthetic system for verification -------------------------------------------------------

def synthetic_torsions(n_frames: int = 6000, n_states: int = 3, n_slow: int = 4,
                       n_fast: int = 40, dwell: int = 400, noise: float = 12.0, seed: int = 0):
    """A trajectory of torsion angles whose metastable states are known by construction.

    `n_slow` angles jump together between `n_states` well-separated values, staying in each for
    ~`dwell` frames; `n_fast` further angles are fast Gaussian noise around fixed values, i.e.
    exactly the nuisance variance TICA is supposed to ignore and RMSD-style clustering is not.
    Returns (angles in degrees, true state per frame).
    """
    rng = np.random.default_rng(seed)
    centres = np.linspace(-150, 150, n_states)
    state, states = 0, []
    while len(states) < n_frames:
        state = int(rng.integers(n_states))
        states += [state] * int(rng.poisson(dwell) + 1)
    states = np.asarray(states[:n_frames])

    slow = centres[states][:, None] + rng.normal(0, noise, size=(n_frames, n_slow))
    fast_centres = rng.uniform(-180, 180, size=n_fast)
    fast = fast_centres[None, :] + rng.normal(0, 3 * noise, size=(n_frames, n_fast))
    angles = np.concatenate([slow, fast], axis=1)
    return ((angles + 180) % 360) - 180, states


def state_purity(assign, truth) -> float:
    """Fraction of frames whose predicted state agrees with the truth under the best matching.

    A cluster labelling is only defined up to a permutation, so each predicted state is mapped to
    the true state it most often coincides with, and the agreement is measured after that.
    """
    assign, truth = np.asarray(assign), np.asarray(truth)
    ok = assign >= 0
    if not ok.any():
        return 0.0
    total = 0
    for m in np.unique(assign[ok]):
        members = truth[ok][assign[ok] == m]
        total += np.bincount(members).max()
    return total / ok.sum()


def write_ensemble(universe, frames, out_pdb, weights=None, total_sampling: int | None = None):
    """Write the selected frames as a multi-model PDB, ready to be a HADDOCK3 `molecules` entry.

    HADDOCK3's rigidbody module computes `sampling_factor = int(sampling / len(models))`, so it
    spends its budget evenly over ensemble members and offers no per-member weight. Passing
    `weights` (e.g. MSM stationary populations) duplicates members so that the even split
    approximates the intended weighting; `total_sampling` caps the number of models written so the
    duplication cannot drive `sampling_factor` below 1.
    """
    frames = list(frames)
    counts = [1] * len(frames)
    if weights is not None:
        w = np.asarray(weights, float)[:len(frames)]
        w = w / w.sum()
        budget = len(frames) if total_sampling is None else min(total_sampling, 4 * len(frames))
        counts = [max(1, int(round(x))) for x in w * budget]
    protein = universe.select_atoms('protein')
    with mda.Writer(str(out_pdb), protein.n_atoms, multiframe=True) as writer:
        for frame, count in zip(frames, counts):
            universe.trajectory[int(frame)]
            for _ in range(count):
                writer.write(protein)
    return {'path': str(out_pdb), 'frames': frames, 'copies': counts,
            'n_models': int(sum(counts)),
            'sampling_factor': None if total_sampling is None
            else int(total_sampling / max(1, sum(counts)))}


def coverage_curve(d: np.ndarray, order: list[int], n_max: int) -> np.ndarray:
    """d_min(N) for N = 1..n_max, given the members in the order the strategy would add them."""
    out, best = [], np.inf
    for i in range(n_max):
        if i < len(order):
            best = min(best, d[order[i]])
        out.append(best)
    return np.asarray(out)
