# biobb_wf_antibody

## What this repo is

BioBB workflow for antibody–antigen docking and CDR-loop dynamics. For one
benchmark complex it runs HADDOCK3 docking, then GROMACS MD of the CDR clusters,
then an AWH free-energy calculation. Everything is driven by one config file.

## Layout

- `src/antibody_wf/`: installable package. `workflow.py` is the entry point
  (stage 0, input download). `stages/haddock.py`, `MD.py` and `AWH.py` are
  stages 1–3. `cdr.py`, `plotting.py` and `utils.py` are helpers.
- `biobb_wf_antibody/config/workflow.yml`: the single config. Steps are named
  `step<stage>_<n>_<name>`, and the structures come from the `reference`,
  `antibody` and `antigen` properties (PDB code + chains + model).
  `.mdp` and HADDOCK configs live beside it.
- `biobb_wf_antibody/notebooks/biobb_antibody.ipynb`: the main interactive
  version. `notebooks/exploration/` holds analysis notebooks (`ab_common.py`
  has the shared helpers).
- `biobb_wf_antibody/array/`: HPC job array. `launch_wf.py --index N` runs
  benchmark complex N. The cluster setup is in `array/README.md`.
- `env/`: pixi/conda environment specs. `output/`: generated results, not
  source.

## Conventions

- Stage names, config sections and `src/` modules stay in sync (`step1_*` ↔
  `haddock.py`, etc.). Change them together.
- Notebook and package share logic. Put reusable code in `src/antibody_wf/`
  and keep the notebook thin.

## Environment

Use the conda environment at `/home/rchaves/miniforge3/envs/biobb_wf_antibody` for
anything that needs the `biobb_*` packages (running the workflow, inspecting the
BioBB APIs, executing the notebook):

```bash
conda run -p /home/rchaves/miniforge3/envs/biobb_wf_antibody python -c "import biobb_common"
```
