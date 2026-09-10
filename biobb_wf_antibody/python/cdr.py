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
