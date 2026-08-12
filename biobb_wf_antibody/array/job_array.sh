#!/bin/bash
#SBATCH --job-name=ab_array
#SBATCH --output=array_logs/ab_%A_%a.out
#SBATCH --error=array_logs/ab_%A_%a.err
#SBATCH --array=0-15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00
#SBATCH --mail-user=ruben.chaves@irbbarcelona.org

# Launch with: sbatch --array=0-15%5 job_array.sh

# Load modules
module gromacs

# Binaries of the modules above, resolved here and not inside python/workflow.yml, so
# the sections of the case keep the launchers of the node the task runs on and not the
# ones the conda environment puts on the PATH
export GMX_BIN=$(which gmx_mpi)
export MPI_BIN=$(which mpirun)

# Activate conda environment
source activate.sh   
mkdir -p array_logs

# One complex of the 'complexes' list of python/workflow.yml per array task, every
# one of them in its own 'results/case_<index>' working directory
python launch_wf.py --index "$SLURM_ARRAY_TASK_ID" --out-dir results \
    --gmx-bin "$GMX_BIN" --mpi-bin "$MPI_BIN" --ncores "$SLURM_CPUS_PER_TASK"

