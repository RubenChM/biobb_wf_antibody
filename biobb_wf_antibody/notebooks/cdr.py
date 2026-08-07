"""CDR-loop bookkeeping: mapping the IMGT regions onto a simulated system and turning
them into the selections the analysis steps need.

The IMGT unique numbering only exists in the ANARCII-renumbered antibody. Every simulated
system (the antibody alone, the antibody-antigen complex) carries its own numbering and
possibly a different chain order, so the regions are matched by *sequence* and expressed as
1-based residue indices. Those indices then feed:

* a **cpptraj atom mask** (``cpptraj_mask``) -- used to select the loops for the clustering;
* a **GROMACS index file** (``make_loop_ndx``) -- still needed by ``gmx_image`` (framework
  fit), ``gmx_rms`` and ``gmx_rmsf``, which take groups and not masks.

The tail of the file keeps the ``gmx cluster`` route (``gmx_cluster_loops``). The notebook no
longer uses it -- ``cpptraj_cluster`` is preferred because the number of clusters can be set
directly instead of tuning an RMSD cutoff -- but it is the only way back to the cutoff-based
GROMOS / Jarvis-Patrick algorithms, so it is kept for reference.
"""

from itertools import permutations
from pathlib import Path

import MDAnalysis as mda
from Bio.Data.IUPACData import protein_letters_3to1

# IMGT unique numbering of the V-DOMAIN:
# https://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html
CDR_RANGES = [(27, 38), (56, 65), (105, 117)]
FRAMEWORK_RANGES = [(1, 26), (39, 55), (66, 104), (118, 129)]

_AA3TO1 = {k.upper(): v for k, v in protein_letters_3to1.items()}


def _seq(residues):
    return ''.join(_AA3TO1.get(r.resname.upper(), 'X') for r in residues)


def imgt_region_ri(target_structure, anarcii_pdb, regions):
    """1-based residue indices of `regions` (IMGT ranges) in `target_structure`.

    The antibody of the target system may carry a different numbering and a different chain order
    than the ANARCII file, so the two are matched by sequence instead of by residue number. Raises
    if no chain order reproduces the target sequence exactly.
    """
    imgt = mda.Universe(str(anarcii_pdb))
    chains = {c: imgt.select_atoms(f'segid {c}').residues
              for c in dict.fromkeys(imgt.select_atoms('protein').atoms.segids)}
    target = mda.Universe(str(target_structure)).select_atoms('protein').residues
    n_antibody = sum(len(v) for v in chains.values())
    target_seq = _seq(target[:n_antibody])

    order = next((p for p in permutations(chains)
                  if ''.join(_seq(chains[c]) for c in p) == target_seq), None)
    if order is None:
        raise RuntimeError(f'cannot match {Path(anarcii_pdb).name} onto '
                           f'{Path(target_structure).name} by sequence')

    ordered = [r for c in order for r in chains[c]]
    hit = lambda n: any(lo <= n <= hi for lo, hi in regions)
    return [i + 1 for i, r in enumerate(ordered) if hit(r.resid)], order, n_antibody


def _runs(indices):
    """[1,2,3,7,8] -> ['1-3', '7-8']"""
    indices, runs, start = sorted(indices), [], None
    for i, n in enumerate(indices):
        if start is None:
            start = n
        if i + 1 == len(indices) or indices[i + 1] != n + 1:
            runs.append(f'{start}-{n}')
            start = None
    return runs


def cpptraj_mask(indices, atoms='@CA'):
    """[1,2,3,7,8] -> '(:1-3,7-8@CA)'

    Cpptraj numbers residues sequentially from 1, so the mask is written from the same residue
    indices used for the GROMACS groups. The parentheses matter: the biobb wrapper turns the mask
    into a `strip` of its negation, and '!:1-3@CA' would not negate the whole expression.
    """
    return f"(:{','.join(_runs(indices))}{atoms})"


def ri_selection(indices):
    """[1,2,3,7,8] -> 'ri 1-3 | ri 7-8'  ('ri' takes a single range per make_ndx command)."""
    return ' | '.join(f'ri {r}' for r in _runs(indices))


def read_ndx(ndx_path):
    """Parse an index file into {group name: [1-based atom numbers]}."""
    groups, current = {}, None
    for line in Path(ndx_path).read_text().splitlines():
        line = line.strip()
        if line.startswith('['):
            current = line.strip('[] ').strip()
            groups[current] = []
        elif current and line:
            groups[current] += [int(x) for x in line.split()]
    return groups


def make_loop_ndx(structure, output_ndx, cdr_ri, fr_ri, binary_path='gmx'):
    """Index file with the Loop / Loop_CA / Framework / Framework_CA groups of the CDR analysis.

    Group numbering follows the 10 default groups (0-9) of a protein-only structure, so the new
    groups land on 10..13. Group 3 is C-alpha.
    """
    from biobb_gromacs.gromacs.make_ndx import make_ndx

    prop = {
        'selection': f'{ri_selection(cdr_ri)}\nname 10 Loop\n'
                     '10 & 3\nname 11 Loop_CA\n'
                     f'{ri_selection(fr_ri)}\nname 12 Framework\n'
                     '12 & 3\nname 13 Framework_CA',
        'binary_path': binary_path
    }
    make_ndx(input_structure_path=str(structure),
             output_ndx_path=str(output_ndx),
             properties=prop)
    return read_ndx(output_ndx)


# ---------------------------------------------------------------------------------------------
# Kept for reference: clustering the loops with `gmx cluster` instead of `cpptraj cluster`.
#
# This is the route the notebook used before, replaced because none of the GROMACS methods take a
# target number of clusters -- gromos and Jarvis-Patrick are driven by an RMSD cutoff, which has
# to be re-tuned for every trajectory and gives no control over the size of the ensemble handed
# to HADDOCK. It is still the only way to those cutoff-based algorithms, and unlike the cpptraj
# wrapper it writes whole-system representatives directly.
# ---------------------------------------------------------------------------------------------

def gmx_cluster_loops(structure, trajectory, index, output_pdb, cutoff=0.13,
                      method='gromos', binary_path='gmx'):
    """Cluster the CDR loops with `gmx cluster` on the Loop_CA group of `index`.

    `gmx cluster` uses a *single* group for both the least-squares fit and the RMSD, so the
    trajectory must already be superposed on the framework and 'nofit' set, otherwise the loops
    are re-fitted onto themselves and their displacement is hidden.
    """
    from biobb_analysis.gromacs.gmx_cluster import gmx_cluster

    prop = {
        'fit_selection': 'Loop_CA',
        'output_selection': 'System',
        'method': method,
        'cutoff': cutoff,
        'nofit': True,
        'binary_path': binary_path
    }
    return gmx_cluster(input_structure_path=str(structure),
                       input_traj_path=str(trajectory),
                       input_index_path=str(index),
                       output_pdb_path=str(output_pdb),
                       properties=prop)
