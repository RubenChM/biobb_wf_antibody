#!/usr/bin/env python3

import time
import argparse
import importlib
from biobb_common.configuration import settings
from biobb_common.tools import file_utils as fu
from biobb_io.api.pdb import pdb
from utils import report_execution, resolve_complex

# The subworkflow modules are named after their index, so they cannot be pulled
# in with a plain import statement
haddock = importlib.import_module('1_haddock')
md = importlib.import_module('2_MD')
awh = importlib.import_module('3_AWH')

# The structures to dock are named by the 'reference', 'antibody' and 'antigen'
# global properties of the configuration file, see utils.resolve_complex for how
# the PDB codes, the chains and the models are derived from them. To dock one of
# the complexes of the benchmark, run array/launch_wf.py with its index: it writes
# a configuration file with those three properties and its own working directory.


def download_pdbs(global_log, global_prop, global_paths, complex_ids):
    """Subworkflow 0: download the input structures from the PDB.

    The PDB code of every structure comes from the selected complex, so it is not
    hardcoded in the configuration file. The subworkflows do not get these paths
    handed over, they pick them up through the 'dependency/step0_*' paths.
    """
    structures = {}
    for name, step, key in [('reference complex', 'step0_1_pdb_reference', 'reference'),
                            ('antibody', 'step0_2_pdb_antibody', 'antibody'),
                            ('antigen', 'step0_3_pdb_antigen', 'antigen')]:
        prop = global_prop[step]
        prop['pdb_code'] = complex_ids[key]['pdb_code']
        global_log.info(f"{step}: Download the {name} structure {prop['pdb_code']} from the PDB")
        pdb(**global_paths[step], properties=prop)
        structures[name] = global_paths[step]['output_pdb_path']
    return structures


def main(config):
    start_time = time.time()
    conf = settings.ConfReader(config)
    global_log, _ = fu.get_logs(path=conf.get_working_dir_path(), light_format=True)
    global_prop = conf.get_prop_dic(global_log=global_log)
    global_paths = conf.get_paths_dic()

    complex_ids = resolve_complex(global_prop['step0_0_pdb_codes'])
    # Subworkflow 0: input structures, shared by all the subworkflows
    download_pdbs(global_log, global_prop, global_paths, complex_ids)
    # Subworkflow 1: antibody-antigen docking with HADDOCK3
    haddock_out = haddock.haddock_workflow(global_log, global_prop, global_paths, complex_ids)
    # Subworkflow 2: free MD of the unbound antibody, its CDR-loop clusters docked
    md_out = md.md_workflow(global_log, global_prop, global_paths, complex_ids)
    # Subworkflow 3: AWH-MD of the best docked complex, its CDR-loop clusters docked.
    awh_out = awh.awh_workflow(global_log, global_prop, global_paths)

    report_execution(global_log, conf, config, start_time, extra_lines=(
        f'Baseline docking: {haddock_out["haddock_wf_data"]}',
        f'MD  CDR ensemble docking: {md_out["haddock_wf_data"]}',
        f'AWH CDR ensemble docking: {awh_out["haddock_wf_data"]}'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Antibody-antigen binding affinity workflow")
    parser.add_argument('--config', '-c', required=True)
    args = parser.parse_args()
    main(args.config)
