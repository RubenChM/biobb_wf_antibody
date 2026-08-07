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


def read_complexes(config_path=TEMPLATE_CONFIG):
    """Return the (reference, antibody, antigen) triplets of the benchmark."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    complexes = config['global_properties']['complexes']
    return [tuple(complex_ids) for complex_ids in complexes]


def write_case_config(index, out_dir):
    """Write the configuration file of one complex and return its path."""
    complexes = read_complexes()
    if not 0 <= index < len(complexes):
        raise SystemExit(f"Index {index} is out of range, only {len(complexes)} complexes defined.")

    case_dir = os.path.abspath(os.path.join(out_dir, 'case_%d' % index))
    os.makedirs(case_dir, exist_ok=True)

    with open(TEMPLATE_CONFIG) as f:
        config = yaml.safe_load(f)

    global_properties = config.setdefault('global_properties', {})
    # The list of the benchmark belongs to the template: the copy names one complex
    reference, antibody, antigen = complexes[index]
    global_properties['working_dir_path'] = case_dir
    global_properties['restart'] = True
    global_properties['reference'] = reference
    global_properties['antibody'] = antibody
    global_properties['antigen'] = antigen

    # 'file:<path>' paths are handed over to the building block as they are
    haddock_run = config.get('step1_12_haddock3_run', {}).get('paths', {})
    if haddock_run.get('haddock_config_path', '').startswith('file:'):
        haddock_run['haddock_config_path'] = 'file:' + os.path.join(
            PYTHON_DIR, haddock_run['haddock_config_path'][len('file:'):])

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
