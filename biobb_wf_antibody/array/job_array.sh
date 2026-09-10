#!/bin/bash
#SBATCH --account=irb93
#SBATCH --qos=gp_resa
#SBATCH --job-name=ab_array
#SBATCH --output=array_logs/ab_%A_%a.out
#SBATCH --error=array_logs/ab_%A_%a.err
#SBATCH --array=0-15
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=112
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=ruben.chaves@irbbarcelona.org

# Launch with: sbatch --array=0-1,4-5,7-14 job_array.sh
# 2,3,6 and 15 are skipped because the are missing aminoacids

echo SLURM_CPUS_PER_TASK $SLURM_CPUS_PER_TASK
echo SLURM_CPUS_ON_NODE $SLURM_CPUS_ON_NODE
echo SLURM_JOB_CPUS_PER_NODE $SLURM_JOB_CPUS_PER_NODE
echo SLURM_NPROCS $SLURM_NPROCS

# https://www.bsc.es/supportkc/docs/MareNostrum5/slurm#special-considerations
# https://www.bsc.es/supportkc/docs/MareNostrum5/Marenostrum5-Applications/GPP/GROMACS#sample-job-script
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
# By default, SLURM sets OMP_NUM_THREADS to cpus-per-task (SLURM_CPUS_PER_TASK)
# https://www.bsc.es/supportkc/docs/MareNostrum5/slurm#other-environment-variables
# Keep that value explicit: each MPI rank must only start the threads allocated
# to its Slurm task.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo OMP_NUM_THREADS $OMP_NUM_THREADS
# Avoid  nthreads cannot be larger than environment variable "NUMEXPR_MAX_THREADS" (64) while calculating haddock restrains
export NUMEXPR_MAX_THREADS="$SLURM_CPUS_PER_TASK"

# Load modules
module purge
module load gcc/14.1.0_binutils241 ucx/1.16.0-gcc openmpi/5.0.5-gcc fftw/3.3.10-gcc-ompi505 boost/1.86.0-gcc-ompi mkl/2023.2.0
module load gromacs/2025.4-gcc-ompi

# Binaries of the modules above, resolved here and not inside python/workflow.yml, so
# the sections of the case keep the launchers of the node the task runs on and not the
# ones the conda environment puts on the PATH
export GMX_BIN=$(which gmx_mpi)
export MPI_BIN=$(which srun)

# Activate conda environment
cd /gpfs/projects/irb93/ruben
source activate.sh   
cd /gpfs/projects/irb93/ruben/ab_wf/array
mkdir -p array_logs

# One complex of the 'complexes' list of python/workflow.yml per array task, every
# one of them in its own 'results/case_<index>' working directory
python launch_wf.py --index "$SLURM_ARRAY_TASK_ID" --out-dir results \
    --gmx-bin "$GMX_BIN" --mpi-bin "$MPI_BIN" --mpi-np "$SLURM_NTASKS" \
    --num-threads-omp "$SLURM_CPUS_PER_TASK" --ncores "$SLURM_CPUS_PER_TASK"
