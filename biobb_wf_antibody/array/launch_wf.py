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

def inject_cluster_runtime(config, gmx_bin=None, mpi_bin=None, mpi_np=None,
                           num_threads_omp=None):
    """Apply the cluster launch settings to every GROMACS simulation step."""
    for name, section in config.items():
        if not isinstance(section, dict):
            continue
        tool = section.get('tool') or name
        properties = section.get('properties') or {}

        if gmx_bin and 'binary_path' in properties:
            properties['binary_path'] = gmx_bin

        if tool not in ('mdrun', 'mdrun_multidir'):
            continue

        properties = section.setdefault('properties', {})
        if mpi_bin:
            properties['mpi_bin'] = mpi_bin
        if mpi_np is not None:
            properties['mpi_np'] = int(mpi_np)
        if num_threads_omp is not None:
            properties['num_threads_omp'] = int(num_threads_omp)


def write_case_config(index, out_dir, gmx_bin=None, mpi_bin=None, ncores=None,
                      mpi_np=None, num_threads_omp=None):
    """Write the configuration file of one complex and return its path."""
    if not 0 <= index < len(COMPLEXES):
        raise SystemExit(f"Index {index} is out of range, only {len(COMPLEXES)} complexes defined.")

    case_dir = os.path.abspath(os.path.join(out_dir, 'case_%d' % index))
    os.makedirs(case_dir, exist_ok=True)

    with open(TEMPLATE_CONFIG) as f:
        config = yaml.safe_load(f)

    global_properties = config.setdefault('global_properties', {})
    global_properties['working_dir_path'] = case_dir
    global_properties['restart'] = True

    # The list of the benchmark belongs to this script: the copy of the template
    # names one complex, through the three identifiers of 'step0_0_pdb_codes'
    reference, antibody, antigen = COMPLEXES[index]
    pdb_codes = config.setdefault('step0_0_pdb_codes', {})
    properties = pdb_codes.setdefault('properties', {})
    properties['reference'] = reference
    properties['antibody'] = antibody
    properties['antigen'] = antigen

    # Drop the 'cfg' and 'mdp' overrides
    for name, section in config.items():
        if not isinstance(section, dict):
            continue
        tool = section.get('tool') or name
        properties = section.get('properties') or {}
        for override, tools in [('cfg', ('haddock3_run',)), ('mdp', ('grompp',))]:
            if override in properties and tool in tools:
                properties.pop(override)
        # The 'ncores' of inputs/haddock_config.cfg is the one of a workstation, on the
        # cluster the docking gets the cores SLURM reserved for the task. A plain key
        # of 'cfg' is a top-level HADDOCK3 parameter, not a section of the run
        if ncores and tool == 'haddock3_run':
            section.setdefault('properties', properties)
            properties.setdefault('cfg', {})['ncores'] = int(ncores)

    # All mdrun steps use the same Slurm allocation. In a conventional MD run the
    # ranks cooperate on one simulation; in mdrun_multidir they are distributed
    # over the AWH walkers. Other GROMACS tools remain single-process.
    inject_cluster_runtime(config, gmx_bin, mpi_bin, mpi_np, num_threads_omp)

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


def main(index, out_dir, dry_run=False, download_only=False, gmx_bin=None, mpi_bin=None,
         ncores=None, mpi_np=None, num_threads_omp=None):
    case_dir, config_path = write_case_config(
        index, out_dir, gmx_bin, mpi_bin, ncores, mpi_np, num_threads_omp)
    if dry_run: return config_path
    # The workflow modules are imported instead of being run in another process, so
    # the task keeps the environment SLURM started it with
    sys.path.insert(0, PYTHON_DIR)
    import workflow
    workflow.main(config_path, download_only=download_only)
    return config_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run one complex of the benchmark")
    parser.add_argument('--index', type=int, default=os.environ.get('SLURM_ARRAY_TASK_ID'),
                        help="index of the complex in the COMPLEXES list of this script, "
                             "defaults to $SLURM_ARRAY_TASK_ID")
    parser.add_argument('--out-dir', default='.',
                        help="folder the 'case_<index>' working directories are created "
                             "in, defaults to the current one")
    parser.add_argument('--dry-run', action='store_true',
                        help="only write the configuration file of the complex")
    parser.add_argument('--download-only', '-d', action='store_true',
                        help="only download the input structures from the PDB")
    parser.add_argument('--gmx-bin', default=os.environ.get('GMX_BIN'),
                        help="GROMACS binary of every GROMACS step, defaults to $GMX_BIN "
                             "and, without it, to the 'gmx_mpi' of the template")
    parser.add_argument('--mpi-bin', default=os.environ.get('MPI_BIN'),
                        help="MPI launcher of every mdrun step, defaults to $MPI_BIN; "
                             "without it, ordinary mdrun steps remain single-rank")
    parser.add_argument('--mpi-np', type=int, default=os.environ.get('SLURM_NTASKS'),
                        help="MPI ranks of every mdrun step, defaults to $SLURM_NTASKS")
    parser.add_argument('--num-threads-omp', type=int,
                        default=os.environ.get('SLURM_CPUS_PER_TASK'),
                        help="OpenMP threads per MPI rank of every mdrun step, defaults "
                             "to $SLURM_CPUS_PER_TASK")
    parser.add_argument('--ncores', type=int, default=os.environ.get('SLURM_CPUS_PER_TASK'),
                        help="cores of the HADDOCK3 dockings, defaults to "
                             "$SLURM_CPUS_PER_TASK and, without it, to the 'ncores' of "
                             "inputs/haddock_config.cfg")

    args = parser.parse_args()
    if args.index is None:
        parser.error("--index is required when $SLURM_ARRAY_TASK_ID is not set")
    main(int(args.index), args.out_dir, args.dry_run, download_only=args.download_only,
         gmx_bin=args.gmx_bin, mpi_bin=args.mpi_bin, ncores=args.ncores,
         mpi_np=args.mpi_np, num_threads_omp=args.num_threads_omp)
