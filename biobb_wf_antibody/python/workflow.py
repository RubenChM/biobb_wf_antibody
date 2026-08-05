#!/usr/bin/env python3

import time
import argparse
import importlib
from biobb_common.configuration import settings
from biobb_common.tools import file_utils as fu
from biobb_io.api.pdb import pdb
from utils import resolve_complex

# The subworkflow modules are named after their index, so they cannot be pulled
# in with a plain import statement
haddock = importlib.import_module('1_haddock')

# The structures to dock are named by the 'reference', 'antibody' and 'antigen'
# global properties of the configuration file, see utils.resolve_complex for how
# the PDB codes, the chains and the models are derived from them. To dock one of
# the complexes of the benchmark, run array/launch_wf.py with its index: it writes
# a configuration file with those three properties and its own working directory.


def get_mdp_dicts(global_prop, global_log):
    """Return the preparation and the production mdp settings.

    They are defined once in the 'prep_mdp' and 'prod_mdp' sections of the
    configuration file and shared by the MD and the AWH subworkflows, so they can
    be handed over as properties to any grompp step.
    """
    prep_mdp_dict = {'mdp': dict(global_prop.get('prep_mdp', {}).get('mdp', {}))}
    prod_mdp_dict = {'mdp': dict(global_prop.get('prod_mdp', {}).get('mdp', {}))}
    report_mdp_dicts(global_log, prep_mdp_dict, prod_mdp_dict)
    return prep_mdp_dict, prod_mdp_dict


def report_mdp_dicts(global_log, prep_mdp_dict, prod_mdp_dict):
    """Log how long the simulations defined by the mdp settings will run"""
    for label, mdp_dict in [('Preparation', prep_mdp_dict), ('Production', prod_mdp_dict)]:
        mdp = mdp_dict['mdp']
        if mdp.get('nsteps') is None:
            global_log.info('  %s simulations have no nsteps set' % label)
            continue
        # 'dt' is given in ps, GROMACS defaults to 1 fs when it is not set
        global_log.info('  %s simulations will run for %g ns'
                        % (label, mdp['nsteps'] * mdp.get('dt', 0.002) / 1000))


def download_workflow(global_log, global_prop, global_paths, complex_ids):
    """Subworkflow 0: download the input structures from the PDB.

    The PDB code of every structure comes from the selected complex, so it is not
    hardcoded in the configuration file. The subworkflows do not get these paths
    handed over, they pick them up through the 'dependency/step0_*' paths.
    """
    structures = {}
    for name, step, key in [('reference complex', 'step0_0_pdb_reference', 'reference'),
                            ('antibody', 'step0_1_pdb_antibody', 'antibody'),
                            ('antigen', 'step0_2_pdb_antigen', 'antigen')]:
        prop = global_prop[step]
        prop['pdb_code'] = complex_ids[key]['pdb_code']
        global_log.info("%s: Download the %s structure %s from the PDB"
                        % (step, name, prop['pdb_code']))
        pdb(**global_paths[step], properties=prop)
        structures[name] = global_paths[step]['output_pdb_path']
    return structures


def main(config):
    start_time = time.time()
    conf = settings.ConfReader(config)
    global_log, _ = fu.get_logs(path=conf.get_working_dir_path(), light_format=True)
    global_prop = conf.get_prop_dic(global_log=global_log)
    global_paths = conf.get_paths_dic()
    # The structures to dock and the length of the simulations are both defined
    # once in the configuration file and shared by every subworkflow
    prep_mdp_dict, prod_mdp_dict = get_mdp_dicts(global_prop, global_log)

    complex_ids = resolve_complex(global_prop['step0_0_pdb_reference'])
    # Subworkflow 0: input structures, shared by all the subworkflows
    download_workflow(global_log, global_prop, global_paths, complex_ids)
    # Subworkflow 1: antibody-antigen docking with HADDOCK3
    haddock_out = haddock.haddock_workflow(global_log, global_prop, global_paths, complex_ids)

    # TODO Subworkflow 2 (2_MD.py, step2_* sections): MD of the CDR clusters,
    # its grompp steps take prep_mdp_dict and prod_mdp_dict
    # TODO Subworkflow 3 (3_AWH.py, step3_* sections): AWH free energy
    # calculation, its grompp steps take prep_mdp_dict and prod_mdp_dict

    elapsed_time = time.time() - start_time
    global_log.info('')
    global_log.info('')
    global_log.info('Execution successful: ')
    global_log.info('  Workflow_path: %s' % conf.get_working_dir_path())
    global_log.info('  Config File: %s' % config)
    global_log.info('  HADDOCK3 workflow data: %s' % haddock_out['haddock_wf_data'])
    global_log.info('')
    global_log.info('Elapsed time: %.1f minutes' % (elapsed_time/60))
    global_log.info('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Antibody-antigen binding affinity workflow")
    parser.add_argument('--config', '-c', required=True)
    args = parser.parse_args()
    main(args.config)
