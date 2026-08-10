#!/usr/bin/env python3

"""Run one complex of the benchmark, as one task of a SLURM job array.

  sbatch --array=0-15 job_array.sh          # the whole list
  python launch_wf.py --index 9             # one complex, without SLURM
  python launch_wf.py --index 9 --dry-run   # only write case_9/workflow.yml
"""

import argparse
import os
import sys
import yaml

# The workflow and its configuration file live next to this script, in python/
PYTHON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python')
PYTHON_DIR = os.path.normpath(PYTHON_DIR)
TEMPLATE_CONFIG = os.path.join(PYTHON_DIR, 'workflow.yml')
COMPLEXES = (
    # Reference         Antibody   Antigen
    ("2VXT_HL:I",	   "2VXU_HL", "1J0S_A"),
    ("2W9E_HL:A",	   "2W9D_HL", "1QM1_A"),
    ("3EOA_LH:I",	   "3EO9_LH", "3F74_A"),
    ("3HMX_LH:AB",	   "3HMW_LH", "1F45_AB"),
    ("3MXW_LH:A",	   "3MXV_LH", "3M1N_A"),
    ("5VPG_CD:A",	   "3RVT_CD", "3F5V_A"),
    ("4DN4_LH:M",	   "4DN3_LH", "1DOL_A"),
    ("4FQI_HL:ABEFCD", "4FQH_HL", "2FK0_ABCDEF"),
    ("4G6J_HL:A",      "4G5Z_HL", "4I1B_A"),
    ("4G6M_HL:A" ,     "4G6K_HL", "4I1B_A"),
    ("4GXU_MN:ABEFCD", "4GXV_HL", "1RUZ_HIJKLM"),
    # Medium
    ("3EO1_AB:CF",     "3EO0_AB", "1TGJ_AB"),
    ("3G6D_LH:A",      "3G6A_LH", "1IK0_A(10)"),
    ("3HI6_XY:B",      "3HI5_HL", "1MJN_A"),
    ("3L5W_LH:I",      "3L7E_LH", "1IK0_A(11)"),
    ("3V6Z_AB:F",      "3V6F_AB", "3KXS_F"),
)

def write_case_config(index, out_dir):
    """Write the configuration file of one complex and return its path."""
    if not 0 <= index < len(COMPLEXES):
        raise SystemExit(f"Index {index} is out of range, only {len(COMPLEXES)} complexes defined.")

    case_dir = os.path.abspath(os.path.join(out_dir, 'case_%d' % index))
    os.makedirs(case_dir, exist_ok=True)

    with open(TEMPLATE_CONFIG) as f:
        config = yaml.safe_load(f)

    global_properties = config.setdefault('global_properties', {})
    # The list of the benchmark belongs to the template: the copy names one complex
    reference, antibody, antigen = COMPLEXES[index]
    global_properties['working_dir_path'] = case_dir
    global_properties['restart'] = True
    global_properties['reference'] = reference
    global_properties['antibody'] = antibody
    global_properties['antigen'] = antigen

    # 'file:<path>' paths are handed over to the building block as they are, so they
    # are relative to the current directory and not to the working one. Every one of
    # them names a file of python/inputs (the HADDOCK3 configuration and the mdp
    # files), and the case runs from wherever SLURM started it, so they are made
    # absolute here.
    for section in config.values():
        if not isinstance(section, dict):
            continue
        paths = section.get('paths') or {}
        for key, value in paths.items():
            if isinstance(value, str) and value.startswith('file:'):
                paths[key] = 'file:' + os.path.join(PYTHON_DIR, value[len('file:'):])

    config_path = os.path.join(case_dir, 'workflow.yml')
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

    print(f'Case {index}: reference {reference}, antibody {antibody}, antigen {antigen}')
    print(f'  Working directory: {case_dir}')
    print(f'  Configuration:     {config_path}')

    return case_dir, config_path


def main(index, out_dir, dry_run=False):
    case_dir, config_path = write_case_config(index, out_dir)
    if dry_run: return config_path
    # The workflow modules are imported instead of being run in another process, so
    # the task keeps the environment SLURM started it with
    sys.path.insert(0, PYTHON_DIR)
    import workflow
    workflow.main(config_path)
    return config_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run one complex of the benchmark")
    parser.add_argument('--index', type=int, default=os.environ.get('SLURM_ARRAY_TASK_ID'),
                        help="index of the complex in the 'complexes' list of "
                             "python/workflow.yml, defaults to $SLURM_ARRAY_TASK_ID")
    parser.add_argument('--out-dir', default='.',
                        help="folder the 'case_<index>' working directories are created "
                             "in, defaults to the current one")
    parser.add_argument('--dry-run', action='store_true',
                        help="only write the configuration file of the complex")
    args = parser.parse_args()
    if args.index is None:
        parser.error("--index is required when $SLURM_ARRAY_TASK_ID is not set")
    main(int(args.index), args.out_dir, args.dry_run)
