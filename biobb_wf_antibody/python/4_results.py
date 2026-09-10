#!/usr/bin/env python3

"""Create the result figures for every completed benchmark case.

This is a manual fourth workflow step.  It scans ``case_<index>`` directories,
plots the HADDOCK score against DockQ for each available docking protocol, and
writes the figures under ``python/4_results/case_<index>`` and joins them into
``python/4_results/results.png`` with four columns per case.
"""

import argparse
import re
import runpy
import sys
from pathlib import Path

import matplotlib
import yaml

from utils import resolve_complex


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = SCRIPT_DIR.parent / "notebooks"
OUTPUT_DIR = SCRIPT_DIR / "results"
sys.path.insert(0, str(NOTEBOOKS_DIR))

from plotting import plot_dockq_vs_score  # noqa: E402


PROTOCOLS = (
    (
        "baseline",
        "Baseline docking",
        (Path("step1_12_haddock3_run/haddock_output"), Path("1_dock/output")),
    ),
    (
        "md",
        "Free-MD CDR ensemble",
        (Path("step2_26_haddock3_run/haddock_output"), Path("2_MD/docking/output")),
    ),
    (
        "awh",
        "AWH CDR ensemble",
        (Path("step3_30_haddock3_run/haddock_output"), Path("3_AWH/docking/output")),
    ),
)
CAPRIEVAL_TABLES = (
    Path("run/08_caprieval/capri_ss.tsv"),
    Path("run/11_caprieval/capri_ss.tsv"),
)
PROTOCOL_COLORS = {
    "baseline": "#1f77b4",
    "md": "#ff7f0e",
    "awh": "#2ca02c",
}


def case_sort_key(case_dir):
    """Sort ``case_<index>`` directories numerically when possible."""
    match = re.fullmatch(r"case_(\d+)", case_dir.name)
    return (0, int(match.group(1))) if match else (1, case_dir.name)


def find_case_dirs(results_dir):
    """Return the case directories directly below ``results_dir``."""
    return sorted(
        (path for path in results_dir.glob("case_*") if path.is_dir()),
        key=case_sort_key,
    )


def complex_title(case_dir):
    """Describe a case using the identifiers in its generated configuration."""
    config_path = case_dir / "workflow.yml"
    if not config_path.is_file():
        return case_dir.name

    with config_path.open() as config_file:
        config = yaml.safe_load(config_file) or {}
    properties = config.get("step0_0_pdb_codes", {}).get("properties", {})
    complex_ids = resolve_complex(properties)
    return (
        f"{case_dir.name}: {complex_ids['antibody']['pdb_code']} + "
        f"{complex_ids['antigen']['pdb_code']} "
        f"(reference {complex_ids['reference']['pdb_code']})"
    )


def find_docking_dir(case_dir, relative_dirs):
    """Find a protocol output in either production or notebook-fixture layout."""
    for relative_dir in relative_dirs:
        docking_dir = case_dir / relative_dir
        if all((docking_dir / table).is_file() for table in CAPRIEVAL_TABLES):
            return docking_dir
    return case_dir / relative_dirs[0]


def create_protocol_figure(case_dir, slug, label, docking_dir, overwrite=False):
    """Create one protocol figure, or skip it when inputs/output require it."""
    output_path = OUTPUT_DIR / case_dir.name / f"{slug}_dockq_vs_score.png"
    if output_path.is_file() and not overwrite:
        print(f"  exists:  {output_path}")
        return output_path

    missing = [table for table in CAPRIEVAL_TABLES if not (docking_dir / table).is_file()]
    if missing:
        print(f"  missing: {label} ({', '.join(map(str, missing))})")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 7))
    try:
        color = PROTOCOL_COLORS[slug]
        plot_dockq_vs_score(docking_dir, ax=axis, colors=(color, color))
        axis.set_title("")
        figure.subplots_adjust(top=0.72, right=0.80)
        figure.suptitle(
            f"DockQ vs HADDOCK score\n{complex_title(case_dir)} · {label}",
            fontsize=12,
            y=0.92,
        )
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)

    print(f"  wrote:   {output_path}")
    return output_path


def create_superposed_figure(case_dir, overwrite=False):
    """Overlay every available protocol when at least two are complete."""
    output_path = OUTPUT_DIR / case_dir.name / "superposed_dockq_vs_score.png"
    if output_path.is_file() and not overwrite:
        print(f"  exists:  {output_path}")
        return output_path

    available_protocols = []
    for slug, label, relative_dirs in PROTOCOLS:
        docking_dir = find_docking_dir(case_dir, relative_dirs)
        if all((docking_dir / table).is_file() for table in CAPRIEVAL_TABLES):
            available_protocols.append((slug, label, docking_dir))

    if len(available_protocols) < 2:
        print("  missing: Superposed protocols (fewer than two completed dockings)")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 7))
    try:
        for slug, label, docking_dir in available_protocols:
            color = PROTOCOL_COLORS[slug]
            plot_dockq_vs_score(
                docking_dir,
                ax=axis,
                colors=(color, color),
                label=label,
                marker_alpha=0.35,
                marker_size=24,
                normalize_density=False,
            )
        axis.set_title("")
        figure.subplots_adjust(top=0.72, right=0.80)
        figure.suptitle(
            f"DockQ vs HADDOCK score\n{complex_title(case_dir)} · All protocols",
            fontsize=12,
            y=0.92,
        )
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)

    print(f"  wrote:   {output_path}")
    return output_path


def create_case_figures(case_dir, overwrite=False):
    """Create every available result figure for one benchmark case."""
    print(case_dir.name)
    (OUTPUT_DIR / case_dir.name).mkdir(parents=True, exist_ok=True)
    images = [
        output_path
        for slug, label, relative_dirs in PROTOCOLS
        if (
            output_path := create_protocol_figure(
                case_dir,
                slug,
                label,
                find_docking_dir(case_dir, relative_dirs),
                overwrite=overwrite,
            )
        )
        is not None
    ]
    superposed_path = create_superposed_figure(case_dir, overwrite=overwrite)
    if superposed_path is not None:
        images.append(superposed_path)
    return images


def main(results_dir, overwrite=False):
    """Create figures for every ``case_*`` folder below ``results_dir``."""
    results_dir = Path(results_dir).expanduser().resolve()
    case_dirs = find_case_dirs(results_dir)
    if not case_dirs:
        raise FileNotFoundError(f"No case directories found under {results_dir}")

    images = []
    for case_dir in case_dirs:
        images.extend(create_case_figures(case_dir, overwrite=overwrite))

    create_gallery = runpy.run_path(str(NOTEBOOKS_DIR / "4_results.py"))["create_gallery"]
    create_gallery(OUTPUT_DIR, case_names=[case_dir.name for case_dir in case_dirs])
    print(f"Found {len(case_dirs)} cases and {len(images)} available result images.")
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create result figures for all antibody benchmark cases"
    )
    parser.add_argument(
        "--results-dir",
        "-d",
        default=SCRIPT_DIR.parent / "array" / "results",
        help="folder containing case_<index> directories (default: array/results)",
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="replace result images that already exist",
    )
    arguments = parser.parse_args()
    main(arguments.results_dir, overwrite=arguments.overwrite)
