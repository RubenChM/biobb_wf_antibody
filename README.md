# biobb_wf_antibody

BioBB workflow for antibody–antigen docking and CDR-loop dynamics. For one
complex it runs a baseline HADDOCK3 docking, a free MD of the unbound antibody
and an AWH-MD of the best docked complex, and docks the CDR-loop ensemble each
simulation samples. One configuration file drives it all:
[`biobb_wf_antibody/config/workflow.yml`](biobb_wf_antibody/config/workflow.yml).
Running the benchmark complexes on a cluster is covered in
[`biobb_wf_antibody/array/README.md`](biobb_wf_antibody/array/README.md).

## Workflow

Each box is a range of the `step<stage>_<n>_<name>` sections of `workflow.yml`.
Labelled edges are what a stage takes from an earlier one. Both ensemble
dockings also reuse the antigen, the reference complex and the restraint tables
of stage 1, so they differ from the baseline run only in the antibody. The
dotted edges lead to `stages/results.py`, which runs separately over the
finished cases. Keep the graph in step with `workflow.yml` and
`src/antibody_wf/stages/` when a stage changes.

```mermaid
flowchart TD
    subgraph S0["Stage 0 · workflow.py"]
        ref["Reference complex<br/>step0_1_pdb_reference"]
        ab["Unbound antibody<br/>step0_2_pdb_antibody"]
        ag["Unbound antigen<br/>step0_3_pdb_antigen"]
    end

    subgraph S1["Stage 1 · haddock.py"]
        prep["Repair and prepare<br/>step1_repair_*<br/>step1_0–1_6"]
        restr["Build restraints<br/>step1_7–1_11"]
        dock1["Baseline docking<br/>step1_12"]
        prep --> restr --> dock1
        prep --> dock1
    end

    subgraph S2["Stage 2 · MD.py"]
        md_setup["Antibody system setup<br/>step2_0–2_13"]
        md_run["Production MD<br/>step2_14–2_18"]
        md_clust["CDR-loop clustering<br/>step2_19–2_23"]
        md_dock["Dock CDR ensemble<br/>step2_24–2_26<br/>stage 1 inputs"]
        md_setup --> md_run --> md_clust --> md_dock
    end

    subgraph S3["Stage 3 · AWH.py"]
        awh_setup["Complex system setup<br/>step3_0–3_15"]
        awh_run["AWH multi-walker MD<br/>step3_16–3_21"]
        awh_clust["CDR-loop clustering<br/>step3_22–3_27"]
        awh_dock["Dock CDR ensemble<br/>step3_28–3_30<br/>stage 1 inputs"]
        awh_setup --> awh_run --> awh_clust --> awh_dock
    end

    results["Compare the three dockings<br/>stages/results.py"]

    ref & ab & ag --> prep
    prep -- "repaired antibody" --> md_setup
    dock1 -- "best model" --> awh_setup
    restr -- "interface" --> awh_run
    md_clust -- "IMGT numbering" --> awh_clust
    dock1 & md_dock & awh_dock -.-> results

    classDef prep fill:#F1EFE8,stroke:#5F5E5A,color:#444441
    classDef md fill:#E1F5EE,stroke:#0F6E56,color:#085041
    classDef dock fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    classDef post fill:#F1EFE8,stroke:#5F5E5A,color:#444441,stroke-dasharray:5 5
    class ref,ab,ag,prep,restr,md_clust,awh_clust prep
    class md_setup,md_run,awh_setup,awh_run md
    class dock1,md_dock,awh_dock dock
    class results post
```

Grey boxes prepare or analyse structures, green ones run GROMACS and purple ones
run HADDOCK3.
