#!/bin/bash
#SBATCH --job-name=ab_array
#SBATCH --output=array_logs/mwf_%A_%a.out
#SBATCH --error=array_logs/mwf_%A_%a.err
#SBATCH --array=0-15
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --time=12:00:00
#SBATCH --mail-user=ruben.chaves@irbbarcelona.org

# Launch with: sbatch --array=0-1036%30 job_array.sh

module load anaconda3
conda activate biobb_wf_antibody

cd "$SLURM_SUBMIT_DIR"
mkdir -p array_logs

# One complex of the 'complexes' list of python/workflow.yml per array task, every
# one of them in its own 'results/case_<index>' working directory
python launch_wf.py --index "$SLURM_ARRAY_TASK_ID" --out-dir results

