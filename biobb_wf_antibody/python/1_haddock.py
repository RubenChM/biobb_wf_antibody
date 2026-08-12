#!/usr/bin/env python3

import time
import argparse
import os
import shutil
from biobb_common.configuration import settings
from biobb_common.tools import file_utils as fu
from biobb_pdb_tools.pdb_tools import biobb_pdb_tidy
from biobb_pdb_tools.pdb_tools import biobb_pdb_selchain
from biobb_pdb_tools.pdb_tools import biobb_pdb_delhetatm
from biobb_pdb_tools.pdb_tools import biobb_pdb_fixinsert
from biobb_pdb_tools.pdb_tools import biobb_pdb_selaltloc
from biobb_pdb_tools.pdb_tools import biobb_pdb_selmodel
from biobb_pdb_tools.pdb_tools import biobb_pdb_keepcoord
from biobb_pdb_tools.pdb_tools import biobb_pdb_reres
from biobb_pdb_tools.pdb_tools import biobb_pdb_chain
from biobb_pdb_tools.pdb_tools import biobb_pdb_chainxseg
from biobb_pdb_tools.pdb_tools.biobb_pdb_merge import biobb_pdb_merge
from biobb_haddock.haddock_restraints.haddock_interface import haddock_interface
from biobb_haddock.haddock_restraints.haddock3_passive_from_active import haddock3_passive_from_active
from biobb_haddock.haddock_restraints.haddock3_actpass_to_ambig import haddock3_actpass_to_ambig
from biobb_haddock.haddock_restraints.haddock3_restrain_bodies import haddock3_restrain_bodies
from biobb_haddock.haddock.haddock3_run import haddock3_run
from utils import (pdb_tools_pipeline, read_interface, report_execution,
                   resolve_complex, zip_pdb_files)


def prepare_antibody(input_pdb_path, output_pdb_path, chains, merge_paths, merge_prop,
                     model=None):
    """Prepare an antibody structure to meet the HADDOCK3 requirements.

    Every chain is extracted and cleaned on its own, the chains are then merged
    back into a single chain A with a continuous residue numbering. The Fc
    region is kept, the chains are tied together later on with unambiguous
    restraints.
    """
    input_pdb_path = os.path.abspath(input_pdb_path)
    step_path = os.path.dirname(output_pdb_path)

    chain_pdb_paths = []
    with fu.change_dir(step_path):
        for ch in [ch.strip() for ch in chains.split(',')]:
            chain_pdb_paths.append(os.path.join(step_path, f'chain_{ch}.pdb'))
            # 0. Extract the requested model + steps
            steps = [(biobb_pdb_selmodel.biobb_pdb_selmodel, {'models': model})] if model else [] 
            steps += [
            (biobb_pdb_tidy.biobb_pdb_tidy,           {'strict': True}),    # 1. Adhere to the format specifications
            (biobb_pdb_selchain.biobb_pdb_selchain,   {'chains': ch}),      # 2. Extract chain
            (biobb_pdb_delhetatm.biobb_pdb_delhetatm, {}),                  # 3. Remove all HETATM records 
            (biobb_pdb_fixinsert.biobb_pdb_fixinsert, {}),                  # 4. Delete insertion codes and shift residue numbering 
            (biobb_pdb_selaltloc.biobb_pdb_selaltloc, {}),                  # 5. Select altloc labels (highest occupancy) 
            (biobb_pdb_keepcoord.biobb_pdb_keepcoord, {}),                  # 6. Remove all non-coordinate records 
            (biobb_pdb_tidy.biobb_pdb_tidy,           {})                   # 7. Adhere to the format specifications
            ]
            pdb_tools_pipeline(input_pdb_path, chain_pdb_paths[-1], steps)

    # Merge the cleaned chains into a single PDB file
    merge_paths['input_file_path'] = zip_pdb_files(chain_pdb_paths, os.path.join(step_path, 'chains.zip'))
    biobb_pdb_merge(**merge_paths, properties=merge_prop)

    with fu.change_dir(step_path):
        steps = [
            (biobb_pdb_reres.biobb_pdb_reres, {'number': 1}),       # 1. Renumber the residues starting from 1
            (biobb_pdb_chain.biobb_pdb_chain, {'chain': 'A'}),      # 2. Modify the chain identifier column
            (biobb_pdb_chainxseg.biobb_pdb_chainxseg, {}),          # 3. Swap the segment identifier for the chain identifier
            (biobb_pdb_tidy.biobb_pdb_tidy, {'strict': True})       # 4. Adhere to the format specifications
        ]
        pdb_tools_pipeline(merge_paths['output_file_path'], output_pdb_path, steps)


def prepare_antigen(input_pdb_path, output_pdb_path, chains, model=None):
    """Prepare an antigen structure to meet the HADDOCK3 requirements.

    The requested chains are extracted and relabelled as a single chain B.
    """
    input_pdb_path = os.path.abspath(input_pdb_path)
    step_path = os.path.dirname(output_pdb_path)

    with fu.change_dir(step_path):
        steps = [(biobb_pdb_selmodel.biobb_pdb_selmodel, {'models': model})] if model else []
        steps += [                               # 0. Keep only the requested model
            (biobb_pdb_tidy.biobb_pdb_tidy, {'strict': True}),            # 1. Adhere to the format specifications
            (biobb_pdb_selchain.biobb_pdb_selchain, {'chains': chains}),  # 2. Extract chains
            (biobb_pdb_chain.biobb_pdb_chain, {'chain': 'B'}),            # 3. Modify the chain identifier column
            (biobb_pdb_chainxseg.biobb_pdb_chainxseg, {}),                # 4. Swap the segment identifier for the chain identifier
            (biobb_pdb_delhetatm.biobb_pdb_delhetatm, {}),                # 5. Remove all HETATM records
            (biobb_pdb_fixinsert.biobb_pdb_fixinsert, {}),                # 6. Delete insertion codes and shift residue numbering
            (biobb_pdb_selaltloc.biobb_pdb_selaltloc, {}),                # 7. Select altloc labels (highest occupancy)
            (biobb_pdb_keepcoord.biobb_pdb_keepcoord, {}),                # 8. Remove all non-coordinate records
            (biobb_pdb_tidy.biobb_pdb_tidy, {'strict': True})             # 9. Adhere to the format specifications
        ]
        pdb_tools_pipeline(input_pdb_path, output_pdb_path, steps)


def merge_structures(input_pdb_paths, output_pdb_path):
    """Merge already prepared structures into a single PDB file"""
    step_path = os.path.dirname(output_pdb_path)

    with fu.change_dir(step_path):
        zip_file_path = zip_pdb_files(input_pdb_paths, os.path.join(step_path, 'structures.zip'))
        steps = [
            (biobb_pdb_merge, {}),                                  # 1. Merge several PDB files into one
            (biobb_pdb_tidy.biobb_pdb_tidy, {'strict': True})       # 2. Adhere to the format specifications
        ]
        pdb_tools_pipeline(zip_file_path, output_pdb_path, steps)


def haddock_workflow(global_log, global_prop, global_paths, complex_ids=None):
    """Subworkflow 1: antibody-antigen docking with HADDOCK3

    The chains to keep from every structure come from the 'reference', 'antibody'
    and 'antigen' global properties, 'complex_ids' is the already resolved
    selection when this subworkflow is run from workflow.py.
    """
    if complex_ids is None:
        complex_ids = resolve_complex(global_prop["step1_0_prepare_antibody"])
    reference, antibody, antigen = (complex_ids['reference'], complex_ids['antibody'],
                                    complex_ids['antigen'])
    global_log.info(f'  Antibody {antibody["pdb_code"]} chains {antibody["chains"]}, '
                    f'antigen {antigen["pdb_code"]} chains {antigen["chains"]}, '
                    f'reference {reference["pdb_code"]} chains '
                    f'{reference["antibody_chains"]} + {reference["antigen_chains"]}')
    for name in ('reference', 'antibody', 'antigen'):
        if complex_ids[name]['model'] is not None:
            global_log.info(f'  Model {complex_ids[name]["model"]} of the {name} entry '
                            f'{complex_ids[name]["pdb_code"]} will be extracted')

    global_log.info("step1_0_prepare_antibody: Prepare the antibody structure")
    paths = global_paths["step1_0_prepare_antibody"]
    prepare_antibody(paths['input_pdb_path'], paths['output_pdb_path'],
                     complex_ids['antibody']['chains'],
                     global_paths["step1_1_biobb_pdb_merge_antibody"],
                     global_prop["step1_1_biobb_pdb_merge_antibody"],
                     model=complex_ids['antibody']['model'])
    antibody_prep = paths['output_pdb_path']

    global_log.info("step1_2_prepare_antigen: Prepare the antigen structure")
    paths = global_paths["step1_2_prepare_antigen"]
    prepare_antigen(paths['input_pdb_path'], paths['output_pdb_path'],
                    complex_ids['antigen']['chains'],
                    model=complex_ids['antigen']['model'])
    antigen_prep = paths['output_pdb_path']

    global_log.info("step1_3_prepare_reference_antibody: Prepare the antibody of the reference complex")
    paths = global_paths["step1_3_prepare_reference_antibody"]
    prepare_antibody(paths['input_pdb_path'], paths['output_pdb_path'],
                     complex_ids['reference']['antibody_chains'],
                     global_paths["step1_4_biobb_pdb_merge_reference_antibody"],
                     global_prop["step1_4_biobb_pdb_merge_reference_antibody"],
                     model=complex_ids['reference']['model'])
    reference_antibody = paths['output_pdb_path']

    global_log.info("step1_5_prepare_reference_antigen: Prepare the antigen of the reference complex")
    paths = global_paths["step1_5_prepare_reference_antigen"]
    prepare_antigen(paths['input_pdb_path'], paths['output_pdb_path'],
                    complex_ids['reference']['antigen_chains'],
                    model=complex_ids['reference']['model'])
    reference_antigen = paths['output_pdb_path']

    global_log.info("step1_6_merge_reference_complex: Merge the antibody and the antigen of the reference complex")
    reference_prep = global_paths["step1_6_merge_reference_complex"]['output_pdb_path']
    merge_structures([reference_antibody, reference_antigen], reference_prep)

    global_log.info("step1_7_haddock_interface: Get the contact residues in the interface of the reference complex")
    paths = global_paths["step1_7_haddock_interface"]
    haddock_interface(**paths, properties=global_prop["step1_7_haddock_interface"])
    # The paratope and the epitope are read from the reference complex interface
    # TODO: align reference sequence with ag and ab to check if the numbering is correct
    # as it can be different depending on the number of missing amino acids in the PDB structure.
    interface = read_interface(paths['output_txt_path'])
    paratope, epitope = interface['A'], interface['B']
    global_log.info(f'  Paratope (antibody, chain A): {", ".join(map(str, paratope))}')
    global_log.info(f'  Epitope  (antigen,  chain B): {", ".join(map(str, epitope))}')

    global_log.info("step1_8_antibody_actpass: Active (paratope) and passive residues of the antibody")
    antibody_actpass = global_paths["step1_8_antibody_actpass"]['output_actpass_path']
    fu.create_dir(os.path.dirname(antibody_actpass))
    with open(antibody_actpass, 'w') as f:
        # The whole paratope is active, the passive residues line is left empty
        f.write(' '.join(map(str, paratope)) + '\n\n')

    global_log.info("step1_9_haddock3_passive_from_active: Passive residues around the antigen epitope")
    paths = global_paths["step1_9_haddock3_passive_from_active"]
    prop = global_prop["step1_9_haddock3_passive_from_active"]
    prop['active_list'] = ','.join(map(str, epitope))
    haddock3_passive_from_active(**paths, properties=prop)

    global_log.info("step1_10_haddock3_actpass_to_ambig: Convert active/passive residues to ambiguous restraints")
    paths = global_paths["step1_10_haddock3_actpass_to_ambig"]
    haddock3_actpass_to_ambig(**paths, properties=global_prop["step1_10_haddock3_actpass_to_ambig"])
    ambig_tbl = paths['output_tbl_path']

    global_log.info("step1_11_haddock3_restrain_bodies: Unambiguous restraints tying the antibody chains together")
    paths = global_paths["step1_11_haddock3_restrain_bodies"]
    haddock3_restrain_bodies(**paths, properties=global_prop["step1_11_haddock3_restrain_bodies"])
    unambig_tbl = paths['output_tbl_path']

    global_log.info("step1_12_haddock3_run: Run the HADDOCK3 antibody-antigen docking")
    paths = global_paths["step1_12_haddock3_run"]
    # Every file referenced by haddock_config.cfg must live in the input folder,
    # the names below are the ones the configuration expects
    fu.create_dir(paths['input_haddock_wf_data'])
    for name, src in [('antibody.pdb', antibody_prep), ('antigen.pdb', antigen_prep),
                      ('reference.pdb', reference_prep), ('ambig.tbl', ambig_tbl),
                      ('unambig.tbl', unambig_tbl)]:
        shutil.copy2(src, os.path.join(paths['input_haddock_wf_data'], name))
    # The configuration file and the properties of this step have to hold the same
    # protocol as the two CDR-loop ensemble runs, otherwise the three are not comparable
    haddock3_run(**paths, properties=global_prop["step1_12_haddock3_run"])


    # The MD (2_MD.py) and the AWH (3_AWH.py) subworkflows dock the CDR-loop
    # ensembles they generate against the same antigen and with the same restraints,
    # and the AWH one simulates the best model of this run, so everything they reuse
    # is handed over here instead of being derived again
    return {
        'antibody_pdb_path': antibody_prep,
        'antigen_pdb_path': antigen_prep,
        'reference_pdb_path': reference_prep,
        'ambig_tbl_path': ambig_tbl,
        'unambig_tbl_path': unambig_tbl,
        'paratope': paratope,
        'epitope': epitope,
        'haddock_wf_data': paths['output_haddock_wf_data'],
    }


def main(config):
    start_time = time.time()
    conf = settings.ConfReader(config)
    global_log, _ = fu.get_logs(path=conf.get_working_dir_path(), light_format=True)
    global_prop = conf.get_prop_dic(global_log=global_log)
    global_paths = conf.get_paths_dic()

    haddock_workflow(global_log, global_prop, global_paths)
    report_execution(global_log, conf, config, start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Antibody-antigen docking with HADDOCK3")
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
