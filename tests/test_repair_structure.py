"""Check which benchmark inputs stay broken after repair_structure, per extra_gap.

A residue gap that survives the repair (C(i)-N(i+1) too long inside a chain) is
what later breaks the GROMACS topology. For every complex the antibody and the
antigen are repaired with increasing 'extra_gap' until no gap is left.

The unrepaired inputs are downloaded as stage 0 does (SEQRES included) into
output/repair_test/inputs. Needs the MODELLER key in MODELLER_KEY or biobb_wf_antibody/config/modeller_key.txt.

    python tests/test_repair_structure.py [--gaps 0 2 4 8] [--cases 0 1 2] [--jobs 4]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'biobb_wf_antibody' / 'array'))

FIX_KNOWN = bool(int(os.environ.get('FIX_KNOWN', 0)))  # keep the coordinates of the known residues

from antibody_wf.utils import backbone_breaks  # noqa: E402,F401  (re-exported for the benchmark test)


def load_key():
    if os.environ.get('MODELLER_KEY'):
        return os.environ['MODELLER_KEY']
    key_file = ROOT / 'biobb_wf_antibody' / 'config' / 'modeller_key.txt'
    return key_file.read_text().strip() if key_file.exists() else None


def download_input(code, model, folder):
    """Fetch the entry like step0_2/3 do: assembly 1 unless a model is selected."""
    from biobb_io.api.pdb import pdb
    path = Path(folder) / f'{code}_{model or "asm"}.pdb'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pdb(output_pdb_path=str(path),
            properties={'pdb_code': code, 'api_id': 'pdb', 'restart': False,
                        'assembly': 1 if model is None else None,
                        'filter': ['ATOM', 'MODEL', 'ENDMDL', 'SEQRES']})
    return path


def repair_case(args):
    index, name, extra_gap, workdir = args
    from antibody_wf import utils
    from launch_wf import COMPLEXES
    code, chains, _, model = utils.parse_identifier(COMPLEXES[index][1 if name == 'antibody' else 2])
    src = download_input(code, model, Path(workdir) / 'inputs')
    out = Path(workdir) / f'case_{index}_{name}_gap{extra_gap}' / 'fixed.pdb'
    props = {'extra_gap': extra_gap, 'fix_known': FIX_KNOWN}
    if load_key():
        props['modeller_key'] = load_key()
    try:
        utils.repair_structure(src, out, chains, model=model, assembly=name == 'antigen',
                               properties=props, check_breaks=False)  # the sweep counts them
        return index, name, extra_gap, backbone_breaks(out), None
    except Exception as exc:  # report, keep sweeping
        return index, name, extra_gap, None, f'{type(exc).__name__}: {exc}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gaps', type=int, nargs='*', default=[0, 1, 2, 4, 6, 8, 10, 12])
    ap.add_argument('--cases', type=int, nargs='*')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--workdir', default=str(ROOT / 'output' / ('repair_test' + ('_fixknown' if FIX_KNOWN else ''))))
    a = ap.parse_args()
    from launch_wf import COMPLEXES
    cases = a.cases if a.cases else range(len(COMPLEXES))
    gaps = sorted(a.gaps)
    jobs = [(i, n, g, a.workdir) for i in cases for n in ('antibody', 'antigen') for g in gaps]
    from antibody_wf import utils
    for i in cases:
        for k in (1, 2):
            code, _, _, model = utils.parse_identifier(COMPLEXES[i][k])
            download_input(code, model, Path(a.workdir) / 'inputs')
    results = {}
    with ProcessPoolExecutor(a.jobs) as pool:
        for i, n, g, breaks, err in pool.map(repair_case, jobs):
            results[(i, n, g)] = (breaks, err)
    print('\ncase name      ' + ' '.join(f'g={g:<3}' for g in gaps) + ' minimal')
    for i in cases:
        for n in ('antibody', 'antigen'):
            row, minimal = [], None
            for g in gaps:
                breaks, err = results[(i, n, g)]
                row.append('ERR ' if err else f'{len(breaks):<4}')
                if minimal is None and breaks == []:
                    minimal = g
            print(f'{i:<4} {n:<9} ' + ' '.join(f'{c:<5}' for c in row), minimal if minimal is not None else 'NONE')
    for (i, n, g), (breaks, err) in sorted(results.items()):
        if err:
            print(f'ERR case {i} {n} gap {g}: {err}')
    last = gaps[-1]
    for i in cases:
        for n in ('antibody', 'antigen'):
            breaks, _ = results[(i, n, last)]
            if breaks:
                print(f'case {i} {n} still broken at gap {last}: {breaks}')


if __name__ == '__main__':
    main()
