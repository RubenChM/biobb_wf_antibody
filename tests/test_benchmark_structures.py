"""Check the repaired structures of every benchmark case against the Zlab benchmark 5.5.

For each case the antibody and the antigen are repaired as stage 1 does and compared
with the benchmark files of the same complex ('r' is the antibody, 'l' the antigen):

* chains: the repaired chain IDs are those of the benchmark unbound structure, none
  missing and none unexpected, and every residue of the benchmark chain is present;
* unbound: the C-alpha RMSD to the benchmark unbound structure is small, it is the
  same crystal structure;
* bound: the C-alpha RMSD to the benchmark bound structure is no worse than the one
  between the benchmark unbound and bound structures, plus a margin.

Needs Modeller (MODELLER_KEY or biobb_wf_antibody/config/modeller_key.txt) and network
the first time: the benchmark (47 MB) is cached in output/benchmark5.5, the repaired
structures are written to output/benchmark_test, delete them to repeat the repair
(REUSE_REPAIRED=1 keeps them between runs, to tune the checks only).

    python -m pytest tests/test_benchmark_structures.py [-k "case11"]
    FIX_KNOWN=1 EXTRA_GAP=0 JOBS=8 python -m pytest tests/test_benchmark_structures.py
"""
import os
import sys
import tarfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'biobb_wf_antibody' / 'array'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from antibody_wf import utils  # noqa: E402
from launch_wf import COMPLEXES  # noqa: E402
from test_repair_structure import repair_case, backbone_breaks  # noqa: E402

BENCHMARK_URL = 'https://zlab.wenglab.org/benchmark/benchmark5.5.tgz'
BENCHMARK_DIR = ROOT / 'output' / 'benchmark5.5'
WORKDIR = ROOT / 'output' / 'benchmark_test'
EXTRA_GAP = int(os.environ.get('EXTRA_GAP', 0))
JOBS = int(os.environ.get('JOBS', 4))
# The benchmark names a complex by a crystal form that is not always the reference's
BENCHMARK_CODE = {'5VPG': '3RVW'}
SIDE = {'antibody': 'r', 'antigen': 'l'}

UNBOUND_RMSD_MAX = 3.0   # A, repaired vs benchmark unbound (same entry, rebuilt loops)
BOUND_MARGIN = 1.5       # A, allowed above the benchmark unbound-vs-bound RMSD


# ---------------------------------------------------------------- benchmark files

def benchmark_code(case):
    code = utils.parse_identifier(COMPLEXES[case][0])[0]
    return BENCHMARK_CODE.get(code, code)


def fetch_benchmark(codes):
    """Extract the structures of 'codes' from the benchmark archive, once."""
    wanted = {f'{c}_{s}_{b}.pdb' for c in codes for s in 'lr' for b in 'ub'}
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    missing = {n for n in wanted if not (BENCHMARK_DIR / n).exists()}
    if not missing:
        return
    archive = BENCHMARK_DIR / 'benchmark5.5.tgz'
    if not archive.exists():
        urllib.request.urlretrieve(BENCHMARK_URL, archive)
    with tarfile.open(archive) as tar:
        for member in tar:
            name = Path(member.name).name
            if name in missing and member.isfile():
                (BENCHMARK_DIR / name).write_bytes(tar.extractfile(member).read())
    still = {n for n in missing if not (BENCHMARK_DIR / n).exists()}
    assert not still, f'Not in the benchmark archive: {sorted(still)}'


def read_ca(pdb_path):
    """{chain: [(one-letter, xyz)]} of the first model, first altloc of every C-alpha."""
    chains = {}
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith('ENDMDL'):
            break
        if line[:6] != 'ATOM  ' or line[12:16].strip() != 'CA' or line[16] not in ' A':
            continue
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        chains.setdefault(line[21], []).append((seq1(line[17:20]) or 'X', xyz))
    return chains


# ---------------------------------------------------------------- comparison

def align(a, b):
    """Index pairs of the residues of sequences a and b matched by a global alignment."""
    aligner = PairwiseAligner(mode='global', match_score=2, mismatch_score=-1,
                              open_gap_score=-3, extend_gap_score=-0.5)
    alignment = aligner.align(a, b)[0]
    return [(i, j) for (a0, a1), (b0, b1) in zip(*alignment.aligned)
            for i, j in zip(range(a0, a1), range(b0, b1))]


def pair_chains(ours, ref, by_id):
    """{our chain: reference chain}, by ID or by the best sequence match."""
    if by_id:
        return {c: c for c in ours if c in ref}
    free, mapping = list(ref), {}
    for chain, residues in ours.items():
        seq = ''.join(r[0] for r in residues)
        scored = [(len(align(seq, ''.join(r[0] for r in ref[c]))), c) for c in free]
        if scored:
            mapping[chain] = max(scored, key=lambda s: s[0])[1]  # first wins on ties
            free.remove(mapping[chain])
    return mapping


def paired_coordinates(ours, ref, mapping):
    """Matched C-alpha coordinates (n, 3) of both structures and the unmatched count of ref."""
    x, y, unmatched = [], [], 0
    for chain, other in mapping.items():
        pairs = align(''.join(r[0] for r in ours[chain]), ''.join(r[0] for r in ref[other]))
        x += [ours[chain][i][1] for i, _ in pairs]
        y += [ref[other][j][1] for _, j in pairs]
        unmatched += len(ref[other]) - len(pairs)
    return np.array(x), np.array(y), unmatched


def rmsd(x, y):
    """C-alpha RMSD after the best superposition (Kabsch)."""
    x, y = x - x.mean(0), y - y.mean(0)
    u, s, vt = np.linalg.svd(x.T @ y)
    s[-1] *= np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt(max((x ** 2).sum() + (y ** 2).sum() - 2 * s.sum(), 0) / len(x)))


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope='session')
def repaired():
    """Repair the antibody and the antigen of every case, once, in parallel."""
    fetch_benchmark({benchmark_code(i) for i in range(len(COMPLEXES))})
    reuse = bool(int(os.environ.get('REUSE_REPAIRED', 0)))
    jobs = [(i, n, EXTRA_GAP, str(WORKDIR)) for i in range(len(COMPLEXES))
            for n in ('antibody', 'antigen')]
    paths = {(i, n): WORKDIR / f'case_{i}_{n}_gap{g}' / 'fixed.pdb' for i, n, g, _ in jobs}
    todo = [j for j in jobs if not (reuse and paths[j[:2]].exists())]
    errors = {}
    with ProcessPoolExecutor(JOBS) as pool:
        for i, n, _, _, err in pool.map(repair_case, todo):
            if err:
                errors[(i, n)] = err
    return paths, errors


@pytest.fixture
def case_structures(repaired, request):
    paths, errors = repaired
    case, name = request.param
    if (case, name) in errors:
        pytest.fail(f'repair_structure failed: {errors[(case, name)]}')
    code = benchmark_code(case)
    side = SIDE[name]
    return (read_ca(paths[(case, name)]), paths[(case, name)],
            read_ca(BENCHMARK_DIR / f'{code}_{side}_u.pdb'),
            read_ca(BENCHMARK_DIR / f'{code}_{side}_b.pdb'))


def parametrize():
    return pytest.mark.parametrize(
        'case_structures', [(i, n) for i in range(len(COMPLEXES)) for n in ('antibody', 'antigen')],
        ids=[f'case{i}-{n}' for i in range(len(COMPLEXES)) for n in ('antibody', 'antigen')],
        indirect=True)


# ---------------------------------------------------------------- tests

@parametrize()
def test_chains_and_residues(case_structures):
    ours, path, unbound, _ = case_structures
    assert sorted(ours) == sorted(unbound), \
        f'chains {sorted(ours)} but the benchmark unbound has {sorted(unbound)}'
    _, _, unmatched = paired_coordinates(ours, unbound, pair_chains(ours, unbound, by_id=True))
    assert unmatched == 0, f'{unmatched} residues of the benchmark unbound are missing'
    assert not backbone_breaks(path)


@parametrize()
def test_rmsd_to_unbound_and_bound(case_structures):
    ours, _, unbound, bound = case_structures
    x, y, _ = paired_coordinates(ours, unbound, pair_chains(ours, unbound, by_id=True))
    to_unbound = rmsd(x, y)
    assert to_unbound < UNBOUND_RMSD_MAX, f'RMSD to the benchmark unbound {to_unbound:.2f} A'

    x, y, _ = paired_coordinates(ours, bound, pair_chains(ours, bound, by_id=False))
    to_bound = rmsd(x, y)
    xu, yb, _ = paired_coordinates(unbound, bound, pair_chains(unbound, bound, by_id=False))
    reference = rmsd(xu, yb)
    print(f'to unbound {to_unbound:.2f}  to bound {to_bound:.2f}  unbound-bound {reference:.2f}')
    assert to_bound < reference + BOUND_MARGIN, \
        f'RMSD to the bound {to_bound:.2f} A vs {reference:.2f} A for the benchmark unbound'
