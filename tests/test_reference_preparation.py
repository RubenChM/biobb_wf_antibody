"""Prepare the reference complex of every benchmark case as stage 1 does (step1_3 to step1_7).

The reference is downloaded like step0_1 (assembly 1 unless a model is selected), the
antibody chains are merged into chain A, the antigen chains into chain B, and the
interface is computed. Then:

* both chains hold atoms and both sides of the interface have contacts, otherwise the
  contact mapping of step1_7 fails (an assembly can split the complex across MODELs);
* the prepared complex is the benchmark bound complex ('r_b' antibody + 'l_b' antigen):
  one superposition of both sides gives a C-alpha RMSD close to 0, so the antigen sits
  where it binds and not against a crystal neighbour (assembly 1 of 3G6D pairs the
  antigen with a symmetry copy of the antibody);
* most of the antibody side of the interface lies in the IMGT CDRs, numbered by ANARCII
  as step2_19 does.

No Modeller needed, network the first time: the references are cached in
output/reference_test/inputs, the benchmark in output/benchmark5.5.

    python -m pytest tests/test_reference_preparation.py [-k "case12"] [-s]
"""
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'biobb_wf_antibody' / 'array'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from antibody_wf import cdr, utils  # noqa: E402
from antibody_wf.stages.haddock import prepare_antibody, prepare_antigen, merge_structures  # noqa: E402
from launch_wf import COMPLEXES  # noqa: E402
from test_repair_structure import download_input  # noqa: E402
from test_benchmark_structures import (BENCHMARK_DIR, align, benchmark_code, fetch_benchmark,  # noqa: E402
                                       pair_chains, paired_coordinates, read_ca, rmsd)

WORKDIR = ROOT / 'output' / 'reference_test'
CONFIG = yaml.safe_load((ROOT / 'biobb_wf_antibody' / 'config' / 'workflow.yml').read_text())

BOUND_RMSD_MAX = 1.0      # A, prepared reference vs benchmark bound complex (same crystal)
# The benchmark names a complex by a crystal form that is not always the reference's
# (5VPG vs 3RVW): the same complex, but not the same coordinates.
OTHER_CRYSTAL_RMSD_MAX = 3.0
CDR_FRACTION_MIN = 0.5    # of the antibody interface residues in the IMGT CDRs


def chain_atoms(pdb_path):
    return Counter(line[21] for line in Path(pdb_path).read_text().splitlines()
                   if line.startswith('ATOM'))


@pytest.fixture(scope='module', params=range(len(COMPLEXES)),
                ids=[f'case{i}' for i in range(len(COMPLEXES))])
def reference(request):
    """Paths, chains and interface of the prepared reference complex of one case."""
    from biobb_haddock.haddock_restraints.haddock_interface import haddock_interface
    identifier = COMPLEXES[request.param][0]
    code, antibody_chains, antigen_chains, model = utils.parse_identifier(identifier)
    src = download_input(code, model, WORKDIR / 'inputs')
    case = WORKDIR / f'case_{request.param}'
    # Only the step1_4 and step1_7 folders are created here, as those steps get them from their
    # configuration properties in the workflow; prepare_* must create their own
    steps = {name: case / name for name in ('step1_3', 'step1_4', 'step1_5', 'step1_6', 'step1_7')}
    for name in ('step1_4', 'step1_7'):
        steps[name].mkdir(parents=True, exist_ok=True)
    restart = {'restart': False}

    antibody = str(steps['step1_3'] / 'reference_antibody.pdb')
    antibody_merged = str(steps['step1_4'] / 'reference_antibody_merged.pdb')
    prepare_antibody(str(src), antibody, antibody_chains, {'output_file_path': antibody_merged},
                     restart, model=model)
    antigen = str(steps['step1_5'] / 'reference_antigen.pdb')
    prepare_antigen(str(src), antigen, antigen_chains, model=model)
    complex_pdb = str(steps['step1_6'] / 'reference_clean.pdb')
    merge_structures([antibody, antigen], complex_pdb)

    interface_txt = str(steps['step1_7'] / 'reference_interface.txt')
    haddock_interface(input_pdb_path=complex_pdb, output_txt_path=interface_txt,
                      properties=dict(CONFIG['step1_7_haddock_interface']['properties'], **restart))
    entry_chain_order = list(dict.fromkeys(line[21] for line in src.read_text().splitlines()
                                           if line.startswith('ATOM')))
    return {'case': request.param, 'identifier': identifier, 'complex': complex_pdb,
            'antibody': antibody, 'antigen': antigen, 'antibody_chains': antibody_chains.split(','),
            'antigen_chains': antigen_chains.split(','), 'entry_chain_order': entry_chain_order,
            'chain_dir': steps['step1_3'], 'interface': utils.read_interface(interface_txt)}


def test_reference_has_antibody_and_antigen(reference):
    identifier, interface = reference['identifier'], reference['interface']
    atoms = chain_atoms(reference['complex'])
    assert set(atoms) == {'A', 'B'}, f'{identifier}: chains {dict(atoms)}, expected A and B'
    assert interface.get('A') and interface.get('B'), \
        f'{identifier}: interface {interface}, expected contacts on chains A and B'


def test_reference_is_benchmark_bound_complex(reference):
    """One superposition of antibody + antigen onto the benchmark bound complex."""
    code = benchmark_code(reference['case'])
    fetch_benchmark([code])
    # The antibody chains keep their own IDs in the per-chain files of step1_3
    ours = {ch: read_ca(reference['chain_dir'] / f'chain_{ch}.pdb')[ch]
            for ch in reference['antibody_chains']}
    ref_r = read_ca(BENCHMARK_DIR / f'{code}_r_b.pdb')
    x, y, _ = paired_coordinates(ours, ref_r, pair_chains(ours, ref_r, by_id=False))
    # The antigen chains are fused into chain B in the order of the entry, so join the
    # benchmark antigen chains in that order; by sequence alone, identical copies (the
    # HA trimers of 4FQI and 4GXU) could be joined in any order
    antigen = read_ca(reference['antigen'])['B']
    ref_l = read_ca(BENCHMARK_DIR / f'{code}_l_b.pdb')
    seq = ''.join(r[0] for r in antigen)
    in_entry = [c for c in reference['entry_chain_order'] if c in reference['antigen_chains']]
    if set(in_entry) == set(ref_l):
        orders = [in_entry]
    else:
        # The benchmark relabelled the chains (the copies of the 4FQI assembly share their
        # IDs): keep the orders that align best and let the superposition pick among them
        scored = [(len(align(seq, ''.join(r[0] for c in o for r in ref_l[c]))), o)
                  for o in permutations(ref_l)]
        best = max(s for s, _ in scored)
        orders = [o for s, o in scored if s == best]

    def superposed(order):
        joined = sum((ref_l[c] for c in order), [])
        pairs = align(seq, ''.join(r[0] for r in joined))
        xx = np.vstack([x, [antigen[i][1] for i, _ in pairs]])
        yy = np.vstack([y, [joined[j][1] for _, j in pairs]])
        return rmsd(xx, yy), len(xx)

    value, n_ca = min(superposed(o) for o in orders)
    limit = BOUND_RMSD_MAX if code == reference['identifier'][:4] else OTHER_CRYSTAL_RMSD_MAX
    print(f"{reference['identifier']} vs {code}_r_b + {code}_l_b: {value:.2f} A over {n_ca} CA")
    assert value < limit, \
        f"{reference['identifier']}: C-alpha RMSD {value:.2f} A to the benchmark bound complex {code}"


def test_paratope_in_cdrs(reference):
    """The antibody side of the reference interface is mostly CDR residues."""
    from biobb_haddock.utils.anarcii import anarcii
    anarcii_pdb = str(Path(reference['antibody']).with_name('reference_antibody_imgt.pdb'))
    # ANARCII numbers the antibody with its own chains (the merged file of step1_4). pdb_merge
    # ends every input with END and ANARCII stops reading at the first one, so drop them
    merged = Path(reference['complex']).parents[1] / 'step1_4' / 'reference_antibody_merged.pdb'
    chains_pdb = merged.with_name('reference_antibody_chains.pdb')
    chains_pdb.write_text(''.join(line for line in merged.read_text().splitlines(keepends=True)
                                  if not line.startswith('END')) + 'END\n')
    anarcii(input_pdb_path=str(chains_pdb), output_pdb_path=anarcii_pdb, properties={'restart': False})
    # Chain A of the reference is renumbered from 1, so its residue number is the index
    cdr_ri, _, _ = cdr.imgt_region_ri(reference['antibody'], anarcii_pdb, cdr.CDR_RANGES)
    paratope = reference['interface']['A']
    fraction = len(set(paratope) & set(cdr_ri)) / len(paratope)
    print(f"{reference['identifier']}: {fraction:.0%} of {len(paratope)} paratope residues in CDRs")
    assert fraction >= CDR_FRACTION_MIN, \
        f"{reference['identifier']}: only {fraction:.0%} of the paratope {paratope} in the CDRs"
