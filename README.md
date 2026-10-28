# biobb_wf_antibody

BioBB workflow for antibody–antigen docking and CDR-loop dynamics. For one
complex it runs a baseline HADDOCK3 docking, a free MD of the unbound antibody
and an AWH-MD of the best docked complex, and docks the CDR-loop ensemble each
simulation samples. One configuration file drives it all:
[`biobb_wf_antibody/config/workflow.yml`](biobb_wf_antibody/config/workflow.yml).
Running the benchmark complexes on a cluster is covered in
[`biobb_wf_antibody/array/README.md`](biobb_wf_antibody/array/README.md).

## Workflow

![Workflow stages, from the PDB download to the comparison of the three dockings](docs/workflow.svg)

Each box is a range of the `step<stage>_<n>_<name>` sections of `workflow.yml`.
The arrows that leave a stage are what a later one takes from it: stage 2 starts
from the repaired antibody, stage 3 from the best model of the baseline docking,
and stage 3 locates the CDRs through the IMGT numbering of stage 2. Two inputs
are not drawn: both ensemble dockings reuse the antigen, the reference complex
and the restraint tables of stage 1, and the AWH interval of stage 3 opens from
the interface of `step1_7`. `stages/results.py` runs separately over the
finished cases.

The graph is drawn by hand in [`docs/workflow.svg`](docs/workflow.svg). Update it
when a stage changes.
