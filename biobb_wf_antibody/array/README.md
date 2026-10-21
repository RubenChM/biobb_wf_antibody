# Cluster instructions
``` shell
# Set the remote path to the cluster: user@<cluster>:/path/to/biobb_wf_antibody/
HPC_PATH=mn5t:/gpfs/projects/irb93/ruben/ab_wf
# Create the portable environment
cd ../../env
pixi lock
./build_haddock3_wheel.sh           # see "haddock3" below; only when its version changes
pixi-pack --create-executable --inject haddock3-*.whl
scp environment.sh $HPC_PATH/

# Sync the workflow code. Only transfers files that changed (size/mtime)
cd ..
rsync -avz biobb_wf_antibody src $HPC_PATH/  --exclude __pycache__ --exclude output --exclude log

# Dowload and sync the PDB starting structures and force-field
for ID in $(seq 0 15); do
    python array/launch_wf.py --index $ID --out-dir ../../output/hpc -d
done

mkdir -p output/ff
curl  -o output/ff/charmm36-jul2017.ff.tgz https://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/charmm36-jul2017.ff.tgz
tar -xvzf output/ff/charmm36-jul2017.ff.tgz -C output/ff/  

rsync -avz output/hpc/  $HPC_PATH/output/
rsync -avz output/ff/   $HPC_PATH/biobb_wf_antibody/ff/

# On cluster, run:
# Unpacks into ./env/, writes ./activate.sh
./environment.sh
# Activate; no conda/pixi needed on the host    
source activate.sh
# Test BioBBs and haddock3 are available
gmx_image --help
haddock3 --version
# Submit the job array
sbatch biobb_wf_antibody/array/job_array.sh
```

## haddock3

The bioconda `haddock_biobb` package ships **no files** — only a post-link script that
runs `pip install haddock3`. conda/mamba run post-link scripts, pixi deliberately does
not, so a packed env has the `biobb_haddock` wrappers but no `haddock3` binary.
`build_haddock3_wheel.sh` builds that wheel locally (PyPI has sdist only) and
`--inject` embeds it in `environment.sh`; nothing is needed on the cluster side.

Rebuild the wheel when the haddock3 version changes, or if `pixi.lock` moves to a
python other than 3.12 (the wheel is tagged `cp312`). It targets glibc 2.14 via the
conda-forge toolchain, and `-march=x86-64-v3` — override with
`HADDOCK_MARCH=x86-64-v4` for AVX-512 on MN5, at the cost of running nowhere older.
