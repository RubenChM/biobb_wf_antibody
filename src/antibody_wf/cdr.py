import re
import shutil
from itertools import permutations
from pathlib import Path

import MDAnalysis as mda
from Bio.Data.IUPACData import protein_letters_3to1
from biobb_analysis.gromacs.gmx_cluster import gmx_cluster

# IMGT unique numbering of the V-DOMAIN:
# https://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html
CDR_RANGES = [(27, 38), (56, 65), (105, 117)]
FRAMEWORK_RANGES = [(1, 26), (39, 55), (66, 104), (118, 129)]

_AA3TO1 = {k.upper(): v for k, v in protein_letters_3to1.items()}


def _seq(residues):
    """Return the one-letter sequence of a list of residues, using 'X' for unknown residues."""
    return ''.join(_AA3TO1.get(r.resname.upper(), 'X') for r in residues)


def imgt_region_ri(target_structure, anarcii_pdb, regions):
    """1-based residue indices of `regions` (IMGT ranges) in `target_structure`.

    The antibody of the target system may carry a different numbering and a different chain order
    than the ANARCII file, so the two are matched by sequence instead of by residue number. Raises
    if no chain order reproduces the target sequence exactly. Indices may repeat when a GRO file
    merges insertion-coded residues under one GROMACS residue index.
    """
    imgt = mda.Universe(str(anarcii_pdb))
    chains = {c: imgt.select_atoms(f'segid {c}').residues
              for c in dict.fromkeys(imgt.select_atoms('protein').atoms.segids)}
    # One C-alpha represents one GROMACS residue. Using Residue objects is
    # unsafe for GRO files because insertion codes are lost and MDAnalysis
    # merges adjacent residues with the same residue number and name.
    target_ca = mda.Universe(str(target_structure)).select_atoms('protein and name CA')
    n_antibody_ca = sum(len(v) for v in chains.values())
    # Keep the exact sequence check while avoiding MDAnalysis merging repeated GRO residues.
    target_seq = _seq(target_ca[:n_antibody_ca])

    order = next((p for p in permutations(chains)
                  if ''.join(_seq(chains[c]) for c in p) == target_seq), None)
    if order is None:
        raise RuntimeError(f'cannot match {Path(anarcii_pdb).name} onto '
                           f'{Path(target_structure).name} by sequence')

    ordered = [r for c in order for r in chains[c]]
    # Convert physical C-alpha positions to the residue indices understood by GROMACS.
    target_ri = target_ca[:n_antibody_ca].resindices + 1
    hit = lambda n: any(lo <= n <= hi for lo, hi in regions)
    region_ri = [int(target_ri[i]) for i, r in enumerate(ordered) if hit(r.resid)]
    return region_ri, order, int(target_ri[-1])


def _runs(indices):
    """[1,2,3,7,8] -> ['1-3', '7-8']"""
    # Repeated GRO residues share an index; make_ndx only needs it once.
    indices, runs, start = sorted(set(indices)), [], None
    for i, n in enumerate(indices):
        if start is None:
            start = n
        if i + 1 == len(indices) or indices[i + 1] != n + 1:
            runs.append(f'{start}-{n}')
            start = None
    return runs


def ri_selection(indices):
    """[1,2,3,7,8] -> 'ri 1-3 | ri 7-8'  ('ri' takes a single range per make_ndx command)."""
    return ' | '.join(f'ri {r}' for r in _runs(indices))


def _cluster_sizes(log_path):
    """Cluster sizes from a gmx cluster log, in the order the representatives are written.

    Only the first line of each cluster starts with its number; long member lists continue on
    lines that start with '|', so they are skipped.
    """
    return [int(m.group(1)) for line in Path(log_path).read_text().splitlines()
            if (m := re.match(r'\s*\d+\s*\|\s*(\d+)', line))]


def _max_rmsd(log_path):
    """Largest pairwise RMSD (nm) gmx cluster reports in its log ('The RMSD ranges from 0 to X nm')"""
    return float(re.search(r'RMSD ranges from \S+ to (\S+) nm', Path(log_path).read_text()).group(1))


def _pdb_models(pdb_path):
    """Split a gmx cluster PDB into one block of lines per representative.

    gmx writes TITLE/REMARK/CRYST1 before each MODEL, so a block ends at ENDMDL. A single cluster
    is written with no MODEL record at all, and is wrapped in one here so pdb_splitmodel finds it.
    """
    blocks, current = [], []
    for line in Path(pdb_path).read_text().splitlines(keepends=True):
        if line.startswith('END') and not line.startswith('ENDMDL'):
            continue
        current.append(line)
        if line.startswith('ENDMDL'):
            blocks.append(current)
            current = []
    if not blocks and any(line.startswith('ATOM') for line in current):
        body = [line for line in current if line[:6] not in ('TITLE ', 'REMARK', 'CRYST1')]
        blocks = [['MODEL        1\n', *body, 'ENDMDL\n']]
    return blocks


def cluster_loops(input_structure_path, input_traj_path, input_index_path, output_pdb_path,
                  properties, min_clusters=20, tolerance=0.001, log=print):
    """Cluster the loop conformations with the largest cutoff that yields `min_clusters` clusters.

    The 'cutoff' of `properties` is where the search starts: if it gives enough clusters it is
    raised, up to the largest pairwise RMSD of the trajectory, otherwise it is lowered, down to
    `tolerance`, bisecting in either direction to a resolution of `tolerance` nm. Every
    trial is a full gmx_cluster run, kept under 'cutoff_search/' next to the output. The output
    holds the `min_clusters` most populated representatives (gromos writes them by decreasing
    size), plus the chosen trial's log as <output>.log. A trajectory that cannot reach
    `min_clusters` (too few distinct frames) keeps every cluster of the smallest cutoff tried.

    Returns the cutoff used and the sizes of the clusters kept.
    """
    output_pdb_path = Path(output_pdb_path)
    search_dir = output_pdb_path.parent / 'cutoff_search'
    trials = {}

    def run(cutoff):
        trial_dir = search_dir / f'{cutoff:.4f}'
        trial_dir.mkdir(parents=True, exist_ok=True)
        pdb, cluster_log = trial_dir / output_pdb_path.name, trial_dir / 'cluster.log'
        gmx_cluster(input_structure_path=str(input_structure_path),
                    input_traj_path=str(input_traj_path),
                    input_index_path=str(input_index_path),
                    output_pdb_path=str(pdb), output_cluster_log_path=str(cluster_log),
                    properties=dict(properties, cutoff=cutoff))
        trials[cutoff] = (pdb, cluster_log, _cluster_sizes(cluster_log))
        log(f'  cutoff {cutoff:.4f} nm -> {len(trials[cutoff][2])} clusters')
        return len(trials[cutoff][2])

    start = properties['cutoff']
    if run(start) >= min_clusters:
        # Enough clusters already: search upwards. Past the largest pairwise RMSD of the
        # trajectory every frame is a neighbour of every other one, so it bounds the search
        lo, hi = start, _max_rmsd(trials[start][1]) + tolerance
    else:
        lo, hi = tolerance, start
        if run(lo) < min_clusters:
            hi = lo
            log(f'  WARNING: {len(trials[lo][2])} clusters at most, fewer than the '
                f'{min_clusters} requested; the trajectory has too few distinct frames. '
                'Keeping all of them')
    if hi > lo:
        # The number of gromos clusters grows as the cutoff shrinks, so this converges on the
        # largest cutoff that still splits the loops into enough clusters
        while hi - lo > tolerance:
            mid = (lo + hi) / 2
            if run(mid) >= min_clusters:
                lo = mid
            else:
                hi = mid
    best = lo

    pdb, cluster_log, sizes = trials[best]
    blocks = _pdb_models(pdb)
    if len(blocks) != len(sizes):
        raise ValueError(f'{pdb} holds {len(blocks)} models but {cluster_log} lists '
                         f'{len(sizes)} clusters')
    output_pdb_path.write_text(''.join(line for block in blocks[:min_clusters] for line in block)
                               + 'END\n')
    shutil.copy2(cluster_log, output_pdb_path.with_suffix('.log'))
    kept = sizes[:min_clusters]
    log(f'  cutoff {best:.4f} nm: {len(sizes)} clusters, keeping the {len(kept)} most populated '
        f'({sum(kept)}/{sum(sizes)} frames, sizes {kept})')
    return best, kept


# ---------------------------------------------------------------------------------------------
# Kept for reference: clustering the loops with `cpptraj cluster` instead of `gmx cluster`.
#
# Its advantage is `algorithm: 'hieragglo'` with `clusters: N`, which sets the size of the
# ensemble directly instead of tuning an RMSD cutoff. Two things to know if it is picked up again:
# the wrapper runs a hardcoded `rms first` on all heavy atoms before clustering, which partly
# undoes the framework superposition (framework Ca RMSD median 0.65->0.69 A on the free MD,
# 0.78->0.91 A on the AWH complex), and `mask` is applied as a `strip`, so the representatives it
# writes only contain the atoms the metric was computed on unless a separate metric mask is used.
# ---------------------------------------------------------------------------------------------

def cpptraj_mask(indices, atoms='@CA'):
    """[1,2,3,7,8] -> '(:1-3,7-8@CA)'

    Cpptraj numbers residues sequentially from 1, so the mask is written from the same residue
    indices used for the GROMACS groups. The parentheses matter: the biobb wrapper turns a strip
    mask into its negation, and '!:1-3@CA' would not negate the whole expression.
    """
    return f"(:{','.join(_runs(indices))}{atoms})"
