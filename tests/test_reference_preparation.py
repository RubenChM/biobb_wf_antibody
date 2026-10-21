"""Prepare the reference complex of every benchmark case as stage 1 does (step1_3 to step1_7).

The reference is downloaded like step0_1 (assembly 1 unless a model is selected), the
antibody chains are merged into chain A, the antigen chains into chain B, and the
interface is computed. Both chains must hold atoms and both sides of the interface
must have contacts, otherwise the contact mapping of step1_7 fails (case 12, 3G6D,
splits the complex across two MODELs of its assembly).

No Modeller needed, network the first time: the references are cached in
output/reference_test/inputs.

    python -m pytest tests/test_reference_preparation.py [-k "case12"]
"""
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'biobb_wf_antibody' / 'array'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from antibody_wf import utils  # noqa: E402
from antibody_wf.stages.haddock import prepare_antibody, prepare_antigen, merge_structures  # noqa: E402
from launch_wf import COMPLEXES  # noqa: E402
from test_repair_structure import download_input  # noqa: E402

WORKDIR = ROOT / 'output' / 'reference_test'
CONFIG = yaml.safe_load((ROOT / 'biobb_wf_antibody' / 'config' / 'workflow.yml').read_text())


def chain_atoms(pdb_path):
    return Counter(line[21] for line in Path(pdb_path).read_text().splitlines()
                   if line.startswith('ATOM'))


@pytest.fixture(params=range(len(COMPLEXES)), ids=[f'case{i}' for i in range(len(COMPLEXES))])
def reference(request):
    """(identifier, merged reference complex, interface) of one case."""
    from biobb_haddock.haddock_restraints.haddock_interface import haddock_interface
    identifier = COMPLEXES[request.param][0]
    code, antibody_chains, antigen_chains, model = utils.parse_identifier(identifier)
    src = download_input(code, model, WORKDIR / 'inputs')
    case = WORKDIR / f'case_{request.param}'
    steps = {name: case / name for name in ('step1_3', 'step1_4', 'step1_5', 'step1_6', 'step1_7')}
    for folder in steps.values():
        folder.mkdir(parents=True, exist_ok=True)
    restart = {'restart': False}

    antibody = str(steps['step1_3'] / 'reference_antibody.pdb')
    prepare_antibody(str(src), antibody, antibody_chains,
                     {'output_file_path': str(steps['step1_4'] / 'reference_antibody_merged.pdb')},
                     restart, model=model)
    antigen = str(steps['step1_5'] / 'reference_antigen.pdb')
    prepare_antigen(str(src), antigen, antigen_chains, model=model)
    complex_pdb = str(steps['step1_6'] / 'reference_clean.pdb')
    merge_structures([antibody, antigen], complex_pdb)

    interface_txt = str(steps['step1_7'] / 'reference_interface.txt')
    haddock_interface(input_pdb_path=complex_pdb, output_txt_path=interface_txt,
                      properties=dict(CONFIG['step1_7_haddock_interface']['properties'], **restart))
    return identifier, complex_pdb, utils.read_interface(interface_txt)


def test_reference_has_antibody_and_antigen(reference):
    identifier, complex_pdb, interface = reference
    atoms = chain_atoms(complex_pdb)
    assert set(atoms) == {'A', 'B'}, f'{identifier}: chains {dict(atoms)}, expected A and B'
    assert interface.get('A') and interface.get('B'), \
        f'{identifier}: interface {interface}, expected contacts on chains A and B'
