#!/usr/bin/env python3

"""Shared helpers of the antibody-antigen workflow.

The structures to dock are named by the 'reference', 'antibody' and 'antigen'
global properties of the configuration file, and every subworkflow needs the same
reading of them, so the parsing of those identifiers lives here instead of in one
of them. Which complex a run docks is decided before the workflow starts, by
array/launch_wf.py, which writes the three identifiers into the configuration file
of the run.

The rest of this module holds the helpers that more than one subworkflow needs:
the pdb_tools pipeline, the reading of the reference interface and the two things
the GROMACS subworkflows share, the CHARMM36 force field and the index selection
built from the CDR/framework definitions of cdr.py.
"""

import os
import sys
import re
import glob
import gzip
import time
import shutil
import zipfile
import tarfile
import warnings
import itertools
import urllib.request
import numpy as np
import MDAnalysis as mda

from pathlib import Path
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1
from MDAnalysis.lib.distances import minimize_vectors


def parse_identifier(identifier):
    """Split a structure identifier into its PDB code, its chains and its model.

    The chains of interest follow the PDB code, the ones after the colon are the
    antigen chains of a reference complex. A trailing '(<n>)' is the model to
    extract from the entry, needed by the NMR ensembles that hold one conformer per
    model:
      '4G6K_HL'     -> ('4G6K', 'H,L', None, '1')
      '4G6M_HL:A'   -> ('4G6M', 'H,L', 'A', '1')
      '1IK0_A(10)'  -> ('1IK0', 'A', None, '10')
    """
    identifier = identifier.strip()
    model_match = re.search(r'\((\d+)\)$', identifier)
    if model_match:
        identifier = identifier[:model_match.start()]
    pdb_code, _, chains = identifier.partition('_')
    before_colon, _, after_colon = chains.partition(':')
    return (pdb_code,
            ','.join(before_colon),
            ','.join(after_colon) if after_colon else None,
            model_match.group(1) if model_match else None)


def parse_seqres(pdb_text):
    """Return {chain_id: one_letter_sequence} from PDB text or SEQRES text.

    Uses the PDB fixed-width fields, preserving a blank chain ID as ' '.
    Other records are ignored; no SEQRES records returns {}. Unknown residue
    names become X (Biopython's seq1 convention). Raises ValueError for missing,
    duplicate or out-of-order records, inconsistent declared lengths, or an
    incomplete sequence. This reads the deposited sequence, including residues
    without coordinates; it does not assign author residue numbers.

    Example: parse_seqres(Path('antibody.pdb').read_text())
    """
    residues, lengths, serials = {}, {}, {}
    for line_number, line in enumerate(pdb_text.splitlines(), 1):
        if line[:6] != 'SEQRES':
            continue
        try:
            serial = int(line[7:10])
            chain = line[11]
            length = int(line[13:17])
        except (ValueError, IndexError) as exc:
            raise ValueError(f'Malformed SEQRES record on line {line_number}') from exc
        if serial != serials.get(chain, 0) + 1:
            raise ValueError(f'Unexpected SEQRES serial {serial} for chain {chain!r}')
        if length < 1 or length != lengths.get(chain, length):
            raise ValueError(f'Inconsistent SEQRES length for chain {chain!r}')
        names = line[19:70].split()
        if not names or any(len(name) != 3 for name in names):
            raise ValueError(f'Malformed SEQRES residues on line {line_number}')
        residues.setdefault(chain, []).extend(names)
        lengths[chain], serials[chain] = length, serial
    for chain, names in residues.items():
        if len(names) != lengths[chain]:
            raise ValueError(f'Chain {chain!r}: SEQRES declares {lengths[chain]} '
                             f'residues but contains {len(names)}')
    return {chain: seq1(''.join(names)) for chain, names in residues.items()}


def write_seqres_fasta(pdb_path, chains, fasta_path):
    """Write selected deposited chain sequences for backbone reconstruction."""
    sequences = parse_seqres(Path(pdb_path).read_text())
    selected = [c.strip() for c in chains.split(',') if c.strip()]
    if not selected or any(c not in sequences for c in selected):
        raise ValueError(f'Missing SEQRES for selected chains in {pdb_path}; '
                         'refresh the downloaded PDB with SEQRES records enabled')
    Path(fasta_path).parent.mkdir(parents=True, exist_ok=True)
    Path(fasta_path).write_text(''.join(f'>{c}\n{sequences[c]}\n' for c in selected))
    return str(fasta_path)


def repair_backbone(input_pdb_path, output_pdb_path, chains, model=None,
                    assembly=False, properties=None):
    """Repair selected original chains before HADDOCK renumbers or fuses them.

    Read canonical sequences from the unfiltered entry; retain chain IDs for
    later MD and AWH recovery. Explicit models take precedence over assemblies.
    """
    from tempfile import TemporaryDirectory
    from biobb_pdb_tools.pdb_tools.biobb_pdb_selmodel import biobb_pdb_selmodel
    from biobb_pdb_tools.pdb_tools.biobb_pdb_mkensemble import biobb_pdb_mkensemble
    from biobb_pdb_tools.pdb_tools.biobb_pdb_selchain import biobb_pdb_selchain
    from biobb_model.model.fix_backbone import fix_backbone

    output = Path(output_pdb_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fasta = write_seqres_fasta(input_pdb_path, chains, output.with_suffix('.fasta'))
    # Temporary preprocessing outputs must not be skipped by restart settings.
    prep = dict(properties or {}, restart=False)
    with TemporaryDirectory(dir=output.parent) as folder:
        selected_model = str(Path(folder) / 'model.pdb')
        selected_chains = str(Path(folder) / 'chains.pdb')
        if assembly and model is None:
            biobb_pdb_mkensemble(input_file_path=str(input_pdb_path),
                                output_file_path=selected_model, properties=prep)
        else:
            biobb_pdb_selmodel(input_file_path=str(input_pdb_path),
                              output_file_path=selected_model,
                              properties=dict(prep, models=model or '1'))
        biobb_pdb_selchain(input_file_path=selected_model, output_file_path=selected_chains,
                          properties=dict(prep, chains=chains))
        result = fix_backbone(input_pdb_path=selected_chains,
                              input_fasta_canonical_sequence_path=fasta,
                              output_pdb_path=str(output), properties=properties)
        if result != 0 or not output.is_file():
            raise RuntimeError(f'Backbone repair failed for {input_pdb_path}')
    return str(output)


def resolve_complex(properties):
    """Return the PDB codes, the chains and the models of the structures to dock.

    They are named by the 'reference', 'antibody' and 'antigen' global properties,
    every one of them a PDB code followed by the chains of interest. Everything is
    derived from those three identifiers, so neither the codes nor the chains nor
    the models are spelled out anywhere else in the configuration file.
    """
    identifiers = {}
    for key in ('reference', 'antibody', 'antigen'):
        identifier = properties.get(key)
        if not identifier:
            raise ValueError(f"The '{key}' global property is not set, it must be a PDB code "
                             "followed by the chains of interest, as in '4G6K_HL'")
        identifiers[key] = str(identifier).strip()

    ref_code, ref_antibody_chains, ref_antigen_chains, ref_model = parse_identifier(identifiers['reference'])
    if not ref_antigen_chains:
        raise ValueError(f"The 'reference' identifier '{identifiers['reference']}' does not "
                         "declare the antigen chains of the complex after a colon, as in "
                         "'4G6M_HL:A'")

    antibody_code, antibody_chains, _, antibody_model = parse_identifier(identifiers['antibody'])
    antigen_code, antigen_chains, _, antigen_model = parse_identifier(identifiers['antigen'])
    for key, chains, example in [('reference', ref_antibody_chains, '4G6M_HL:A'),
                                 ('antibody', antibody_chains, '4G6K_HL'),
                                 ('antigen', antigen_chains, '4I1B_A')]:
        if not chains:
            raise ValueError(f"The '{key}' identifier '{identifiers[key]}' does not declare "
                             f"any chain after its PDB code, as in '{example}'")

    return {
        'reference': {'pdb_code': ref_code,
                      'antibody_chains': ref_antibody_chains,
                      'antigen_chains': ref_antigen_chains,
                      'model': ref_model},
        'antibody': {'pdb_code': antibody_code,
                     'chains': antibody_chains,
                     'model': antibody_model},
        'antigen': {'pdb_code': antigen_code,
                    'chains': antigen_chains,
                    'model': antigen_model},
    }


# ============================================================================
# Logging
# ============================================================================


def report_execution(global_log, conf, config, start_time, extra_lines=()):
    """Log the closing summary of a run.

    Every subworkflow ends with the same block, and so does the whole workflow, which
    adds a line per result of its own through 'extra_lines'. 'start_time' is the
    time.time() the run was started at.
    """
    elapsed_time = time.time() - start_time
    global_log.info('')
    global_log.info('')
    global_log.info('Execution successful: ')
    global_log.info(f'  Workflow_path: {conf.get_working_dir_path()}')
    global_log.info(f'  Config File: {config}')
    for line in extra_lines:
        global_log.info(f'  {line}')
    global_log.info('')
    global_log.info(f'Elapsed time: {elapsed_time/60:.1f} minutes')
    global_log.info('')


# ============================================================================
# pdb_tools
# ============================================================================


def pdb_tools_pipeline(inp_file, out_file, steps):
    """Helper function to concatenate calls to pdb_tools"""
    tmp_file = inp_file
    for step, props in steps:
        # Apply each step in the pipeline
        step(input_file_path=tmp_file, output_file_path=out_file, properties=props)
        tmp_file = 'tmp.pdb'
        os.rename(out_file, tmp_file)
    os.rename(tmp_file, out_file)


def remove_model_lines(pdb_path):
    """Remove MODEL and ENDMDL lines."""
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    with open(pdb_path, 'w') as f:
        for line in lines:
            if not line.startswith('MODEL') and not line.startswith('ENDMDL'):
                f.write(line)


def positional_alignment_score(ref_res, target_res, alignment):
    pairs = [
        (ref_res[i], target_res[j])
        for i, j in alignment.indices.T
        if i >= 0 and j >= 0
    ]

    if len(pairs) < 3:
        return float("inf")

    ref_xyz = np.array([
        residue.atoms.select_atoms("name CA").positions[0]
        for residue, _ in pairs
    ])
    target_xyz = np.array([
        residue.atoms.select_atoms("name CA").positions[0]
        for _, residue in pairs
    ])

    # Kabsch superposition
    ref_center = ref_xyz.mean(axis=0)
    target_center = target_xyz.mean(axis=0)
    ref_centered = ref_xyz - ref_center
    target_centered = target_xyz - target_center

    covariance = target_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T

    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T

    fitted = target_centered @ rotation
    distances = np.linalg.norm(fitted - ref_centered, axis=1)

    # Robust score: avoid one flexible loop dominating the decision
    return np.median(distances) + 0.25 * np.percentile(distances, 90)


def map_contact_residues(reference_pdb, target_pdb, chain, contacts, poslaign=False):
    """Align observed residues and map reference contacts to cleaned PDB numbers."""
    selection = f"protein and chainID {chain}"
    ref_res = mda.Universe(reference_pdb).select_atoms(selection).residues
    target_res = mda.Universe(target_pdb).select_atoms(selection).residues
    if not len(ref_res) or not len(target_res):
        raise ValueError(f"Chain {chain}: no protein residues to align")
    for residues in (ref_res, target_res):
        if len(set(residues.resids)) != len(residues) or any(residues.icodes):
            raise ValueError("Use cleaned PDBs with unique residue numbers and no insertion codes")
    contact_ids = [int(resid) for resid in contacts.replace(',', ' ').split()]
    unknown = sorted(set(contact_ids) - set(ref_res.resids))
    if unknown:
        raise ValueError(f"Chain {chain}: contacts absent from reference PDB: {unknown}; regenerate the interface")

    # Align coordinate sequences: missing residues become gaps, not numbering offsets.
    aligner = PairwiseAligner(
        mode='global', match_score=2, mismatch_score=-1,
        open_gap_score=-10, extend_gap_score=-0.5,
    )
    alignments = aligner.align(
        ref_res.sequence(format='string'), target_res.sequence(format='string'),
    )
    print(f"Chain {chain}: {len(alignments)} equivalent sequence alignments found")
    if len(alignments) > 1000:
        raise ValueError(f"Chain {chain}: too many equivalent alignments; select corresponding chains first")
    contact_maps = []
    for alignment in alignments:
        # Alignment indices are zero-based sequence positions; HADDOCK needs PDB resids.
        residue_map = {
            int(ref_res[i].resid): int(target_res[j].resid)
            for i, j in alignment.indices.T if i >= 0 and j >= 0
        }
        contact_maps.append({resid: residue_map.get(resid) for resid in contact_ids})
    ambiguous = [resid for resid in contact_ids
                     if len({mapping[resid] for mapping in contact_maps}) > 1]
    # If more than one alignment, choose the one with best positional alignment
    if ambiguous:
        if poslaign:
            scores = [
                positional_alignment_score(ref_res, target_res, alignment)
                for alignment in alignments
            ]

            best_order = np.argsort(scores)
            best = best_order[0]

            if len(best_order) > 1:
                margin = scores[best_order[1]] - scores[best]
                if margin < 0.05:
                    raise ValueError(
                        f"Chain {chain}: sequence alignments remain structurally ambiguous "
                        f"(scores={scores[:5]})"
                    )

            contact_map = contact_maps[best]
        else:
            raise ValueError(f"Chain {chain}: equally scoring alignments disagree on contacts {ambiguous}; "
                         "select corresponding chains before mapping")
    else:
        contact_map = contact_maps[0]
    missing = [resid for resid, target in contact_map.items() if target is None]
    if missing:
        warnings.warn(f"Chain {chain}: reference contacts missing in target and omitted: {missing}")
    mapped_contacts = sorted({resid for resid in contact_map.values() if resid is not None})
    if not mapped_contacts:
        raise ValueError(f"Chain {chain}: no reference contacts could be mapped to the target")
    return ', '.join(map(str, mapped_contacts)), alignments[0], contact_map


def read_interface(interface_txt_path):
    """Read the residues of each side of the interface reported by haddock_interface.

    The report has one 'Chain <id>: [<residue>, ...]' line per chain.
    """
    interface = {}
    with open(interface_txt_path) as f:
        for line in f:
            if not line.strip().startswith('Chain'):
                continue
            chain, residues = line.split(':', 1)
            interface[chain.split()[1]] = [int(res) for res in re.findall(r'\d+', residues)]
    return interface


# ============================================================================
# HADDOCK3 results
# ============================================================================


def haddock_best_model(haddock_wf_data, output_pdb_path=None, run_dir='run'):
    """Path of the best model of a finished HADDOCK3 run.

    It is the first model of the first cluster written by the last seletopclusts
    stage. The stages are numbered by HADDOCK3 according to their position in
    haddock_config.cfg, so the number is not hardcoded here.

    HADDOCK3 gzips the structures it writes unless its 'clean' parameter is turned
    off, and it is on by default, so the model usually comes as a '.pdb.gz'. It is
    then decompressed into 'output_pdb_path', as the building blocks downstream read
    a plain PDB file, and the run directory is left as HADDOCK3 wrote it.
    """
    stage_pattern = os.path.join(haddock_wf_data, run_dir, '*_seletopclusts')
    models = (glob.glob(os.path.join(stage_pattern, 'cluster_1_model_1.pdb'))
              + glob.glob(os.path.join(stage_pattern, 'cluster_1_model_1.pdb.gz')))
    if not models:
        raise FileNotFoundError(f"No 'cluster_1_model_1.pdb[.gz]' found under {stage_pattern}, "
                                "the HADDOCK3 run did not reach its seletopclusts stage")
    # The last stage is the most refined one, its number is the largest. An already
    # decompressed model wins over the gzipped one of the same stage
    best = max(models, key=lambda path: (int(os.path.basename(os.path.dirname(path)).split('_')[0]),
                                         not path.endswith('.gz')))
    if not best.endswith('.gz'):
        return best

    if not output_pdb_path:
        raise ValueError(f"{best} is gzipped and no 'output_pdb_path' was given to "
                         "decompress it into")
    os.makedirs(os.path.dirname(os.path.abspath(output_pdb_path)), exist_ok=True)
    with gzip.open(best, 'rt') as compressed, open(output_pdb_path, 'w') as pdb_file:
        shutil.copyfileobj(compressed, pdb_file)
    return output_pdb_path


def recover_chain_ids(
        haddock_pdb_path, antibody_pdb_path, antibody_chains,
        antigen_pdb_path, antigen_chains, output_pdb_path
        ):
    """Restore chains fused into HADDOCK A (antibody) and optionally B (antigen).

    Match selected reference chains by residue sequence, independent of their
    order in the entry. Identical antigen subunits follow the requested order.
    Conflicting antigen IDs get unused IDs with a warning. Return the original
    antibody chain order. Both antigen arguments must be supplied together.
    """
    docked = mda.Universe(haddock_pdb_path)
    # Capture both groups before changing any IDs.
    groups = {c: docked.select_atoms(f'protein and chainID {c}') for c in ('A', 'B')}

    def match(reference_path, requested, fused_id, label):
        ids = tuple(c.strip() for c in requested.split(',') if c.strip())
        if not ids or any(len(c) != 1 for c in ids) or len(set(ids)) != len(ids):
            raise ValueError(f'Unique one-character {label} chain identifiers are required')
        reference = mda.Universe(str(reference_path)).select_atoms('protein')
        sequences = {c: tuple(reference[reference.chainIDs == c].residues.resnames)
                     for c in ids}
        if not all(sequences.values()):
            raise ValueError(f'Missing selected {label} chains in {reference_path}')
        fused_sequence = tuple(groups[fused_id].residues.resnames)
        orders = [order for order in itertools.permutations(ids)
                  if tuple(r for c in order for r in sequences[c]) == fused_sequence]
        if not orders:
            raise ValueError(f'Fused {label} chain {fused_id} does not match selected chains')
        distinct = {tuple(sequences[c] for c in order) for order in orders}
        if len(distinct) > 1 or (label == 'antibody' and len(orders) > 1):
            raise ValueError(f'The order of the original {label} chains is ambiguous')
        return orders[0], sequences

    matches = {'A': match(antibody_pdb_path, antibody_chains, 'A', 'antibody'),
               'B': match(antigen_pdb_path, antigen_chains, 'B', 'antigen')
    }
    antibody_order = matches['A'][0]
    occupied = set(docked.atoms.chainIDs) | {c for order, _ in matches.values() for c in order}

    def unused_id():
        chain = next((c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz'
                      if c not in occupied), None)
        if chain is None:
            raise ValueError('No unused chain ID available for the antigen')
        occupied.add(chain)
        return chain

    for fused_id, (order, sequences) in matches.items():
        offset = 0
        for chain in order:
            output_chain = chain
            if fused_id == 'B' and chain in antibody_order:
                output_chain = unused_id()
                warnings.warn(f'Antigen chain {chain} conflicts with an antibody chain; '
                              f'using {output_chain} in the topology PDB', stacklevel=2)
            end = offset + len(sequences[chain])
            groups[fused_id].residues[offset:end].atoms.chainIDs = output_chain
            offset = end
    if 'B' not in matches and antibody_order[-1] == 'B':
        groups['B'].chainIDs = unused_id()

    Path(output_pdb_path).parent.mkdir(parents=True, exist_ok=True)
    with mda.Writer(str(output_pdb_path), docked.atoms.n_atoms, reindex=False) as writer:
        writer.write(docked.atoms)
    return antibody_order


# ============================================================================
# CHARMM36 force field
# ============================================================================


def ensure_force_field(ff_dir, url, force_field):
    """Download and extract a GROMACS force field, and return the GMXLIB directory.

    'gmx_lib' has to point at the directory that *contains* the '<force_field>.ff'
    one, and every pdb2gmx and grompp step of the GROMACS subworkflows is given the
    same value. Nothing is downloaded when the force field is already there, so
    restarting a run does not fetch it again.
    """
    ff_dir = os.path.abspath(ff_dir)
    ff_path = os.path.join(ff_dir, f'{force_field}.ff')
    if os.path.isdir(ff_path):
        return ff_dir

    os.makedirs(ff_dir, exist_ok=True)
    tgz_path = os.path.join(ff_dir, f'{force_field}.ff.tgz')
    if not os.path.isfile(tgz_path):
        urllib.request.urlretrieve(url, tgz_path)
    with tarfile.open(tgz_path) as tar:
        tar.extractall(ff_dir)
    if not os.path.isdir(ff_path):
        raise FileNotFoundError(f"{tgz_path} does not hold a '{force_field}.ff' directory")
    return ff_dir


# ============================================================================
# CDR and framework regions
# ============================================================================

def cdr_ndx_selection(cdr_ri, fr_ri, ri_selection, antibody_res=None):
    """make_ndx selection building the Loop / Framework groups of a system.

    Group 3 of the default groups is the C-alpha one, so intersecting the regions
    with it gives the Loop_CA and Framework_CA groups the framework fit and the
    clustering run on. 'antibody_res' adds the group holding the antibody alone,
    which the AWH subworkflow needs to keep the antigen out of the ensemble written
    by the clustering.
    """
    selection = (f'{ri_selection(cdr_ri)}\nname 10 Loop\n'
                 '10 & 3\nname 11 Loop_CA\n'
                 f'{ri_selection(fr_ri)}\nname 12 Framework\n'
                 '12 & 3\nname 13 Framework_CA')
    if antibody_res is not None:
        selection += f'\nri 1-{antibody_res}\nname 14 Antibody'
    return selection


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


def check_ndx_groups(global_log, structure_path, ndx_path, expected):
    """Check that the index groups landed on the atoms they are meant to.

    An index file is a list of absolute atom numbers, so a group built from the
    wrong structure resolves to a plausible-looking but wrong selection instead of
    failing. 'expected' gives the number of atoms every group must have.
    """
    groups = read_ndx(ndx_path)
    universe = mda.Universe(str(structure_path))
    for name, n_atoms in expected.items():
        selection = universe.atoms[np.array(groups[name]) - 1]
        global_log.info(f'  [{name}] {selection.n_atoms} atoms / '
                        f'{selection.n_residues} residues, '
                        f'C-alpha only: {set(selection.names) == {"CA"}}')
        if selection.n_atoms != n_atoms:
            raise ValueError(f'{name}: expected {n_atoms} atoms, got {selection.n_atoms}')
        if set(selection.names) != {'CA'}:
            raise ValueError(f'{name}: not all the selected atoms are C-alpha')
    return groups


# ============================================================================
# AWH pulling
# ============================================================================

def pull_group_com(atoms, box, pbcatom=0):
    """Return the COM using GROMACS's pull-group PBC reference atom.

    With pull-groupN-pbcatom=0, GROMACS uses the middle atom in index order
    (the lower middle for an even-sized group). Place each atom in its nearest
    periodic image around that reference *before* averaging with the masses.
    This also handles pull groups containing separately wrapped molecules.

    ``atoms`` must have the same order and masses as the pull group; ``box`` is
    the MDAnalysis unit cell in angstrom/degrees. No extra pull weights are
    supported. ``pbcatom`` is a one-based system atom index (zero
    selects GROMACS's default). The input coordinates are not modified.
    """
    if not len(atoms):
        raise ValueError('Cannot calculate the COM of an empty pull group')
    indices = np.flatnonzero(atoms.indices == pbcatom - 1) if pbcatom else [(len(atoms) - 1) // 2]
    if len(indices) != 1:
        raise ValueError(f'PBC reference atom {pbcatom} is not in the pull group')
    reference = atoms.positions[indices[0]]
    positions = reference + minimize_vectors(atoms.positions - reference, box)
    return np.average(positions, axis=0, weights=atoms.masses)


def select_pull_pbcatom(atoms, reference, box):
    """Map a central CA from a whole, unwrapped docking group onto its MD group.

    Residue order must be unchanged by pdb2gmx. Check that imaging around the
    selected atom does not split neighbouring CAs from the reference structure.
    No coordinates are modified. Returned atom numbers are system-wide, 1-based.
    """
    from MDAnalysis.lib.distances import self_distance_array

    ca = atoms.select_atoms('name CA')
    ref_ca = reference.select_atoms('name CA')
    if len(ca) != len(atoms.residues) or len(ref_ca) != len(ca):
        raise ValueError('Pull reference and MD group must have one CA per matching residue')
    # CA geometry avoids hydrogen/mass differences between the docking and MD inputs.
    central = np.argmin(np.linalg.norm(ref_ca.positions - ref_ca.center_of_geometry(), axis=1))
    pbcatom = int(ca.indices[central] + 1)
    origin = ca.positions[central]
    imaged = origin + minimize_vectors(ca.positions - origin, box)
    neighbours = self_distance_array(ref_ca.positions) < 8.0
    direct = self_distance_array(imaged)
    periodic = self_distance_array(ca.positions, box=box)
    if np.any((direct - periodic)[neighbours] > 1.0):
        raise ValueError('Pull group cannot be imaged whole around its central reference '
                         'atom. Inspect the structure and enlarge/re-equilibrate the box.')
    return pbcatom


def validate_awh_interval(box, start, end, margin=0.1):
    """Validate a 3D distance interval in nm with a 0.1 nm safety margin.

    Use the shortest lattice vector, including combinations in triclinic cells.
    This is conservative for non-reduced cells; box dimensions are in angstrom.
    """
    from itertools import product
    from MDAnalysis.lib.mdamath import triclinic_vectors

    vectors = triclinic_vectors(box)
    shifts = np.array([v for v in product((-1, 0, 1), repeat=3) if any(v)])
    limit = 0.49 * np.linalg.norm(shifts @ vectors, axis=1).min() / 10
    if not np.isfinite([start, end, limit]).all() or not 0 <= start < end < limit - margin:
        raise ValueError(f'AWH interval {start:.3f}-{end:.3f} nm must be below '
                         f'{limit:.3f} nm with {margin:.2f} nm margin. '
                         'Check PBC references or enlarge and re-equilibrate the box; '
                         'do not clip the sampling interval.')
    return limit
