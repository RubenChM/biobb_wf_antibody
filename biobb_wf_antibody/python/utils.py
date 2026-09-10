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
            model_match.group(1) if model_match else '1')


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


def zip_pdb_files(pdb_paths, zip_file_path):
    """Join PDB files in the order expected by ``pdb_merge``.

    ``biobb_pdb_merge`` sorts the archive members by filename before merging
    them. Prefix each basename with its position so the caller's order is not
    changed by descriptive filenames such as ``chain_L.pdb`` and
    ``chain_H.pdb``.
    """
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for index, pdb_path in enumerate(pdb_paths):
            basename = os.path.basename(pdb_path)
            zipf.write(pdb_path, arcname=f'{index:04d}_{basename}')
    return zip_file_path


def map_contact_residues(reference_pdb, target_pdb, chain, contacts):
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
    if ambiguous:
        raise ValueError(f"Chain {chain}: equally scoring alignments disagree on contacts {ambiguous}; "
                         "select corresponding chains before mapping")
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


def recover_antibody_chain_ids(haddock_pdb_path, original_antibody_pdb_path,
                               output_pdb_path, antibody_chains):
    """Recover the antibody chain IDs that HADDOCK fuses into chain A.

    Only the requested chains are read from the original entry. Their residue-name
    sequences are matched against HADDOCK chain A, so unrelated chains in the
    original PDB do not affect the indices and the fused-chain order is recovered.
    If the recovered antibody ends in chain B, move the topology-only antigen away
    from B so that GROMACS still sees a chain transition.
    """
    chain_ids = tuple(chain.strip() for chain in antibody_chains.split(',')
                      if chain.strip())
    if not chain_ids or any(len(chain) != 1 for chain in chain_ids):
        raise ValueError('One-character antibody chain identifiers are required, '
                         f'got {antibody_chains!r}')
    if len(set(chain_ids)) != len(chain_ids):
        raise ValueError(f'Antibody chain identifiers are not unique: {antibody_chains!r}')

    original = mda.Universe(original_antibody_pdb_path).select_atoms('protein')
    # Ignore unrelated chains in the downloaded entry (for example, I and M in
    # 2VXU); only the chains named by the antibody identifier define boundaries.
    sequences = {}
    for chain in chain_ids:
        chain_atoms = original[original.chainIDs == chain]
        if not chain_atoms.n_residues:
            raise ValueError(f'Original antibody {original_antibody_pdb_path} has no '
                             f'protein residues in chain {chain}')
        sequences[chain] = tuple(chain_atoms.residues.resnames)

    docked_u = mda.Universe(haddock_pdb_path)
    fused_antibody = docked_u.select_atoms('protein and chainID A')
    # Capture the original antigen atoms before any antibody residues are renamed
    # to B, otherwise a later ``chainID B`` selection could include both molecules.
    antigen = docked_u.select_atoms('chainID B')
    if not fused_antibody.n_residues:
        raise ValueError(f'Docked complex {haddock_pdb_path} has no protein chain A')
    fused_sequence = tuple(fused_antibody.residues.resnames)

    # The merge step can change chain order, so find the order whose concatenated
    # sequences reproduce the complete fused antibody rather than assuming it.
    candidate_orders = []
    for order in itertools.permutations(chain_ids):
        sequence = tuple(resname for chain in order for resname in sequences[chain])
        if sequence == fused_sequence:
            candidate_orders.append(order)
    if not candidate_orders:
        lengths = ', '.join(f'{chain}={len(sequences[chain])}' for chain in chain_ids)
        raise ValueError(f'Fused antibody chain A in {haddock_pdb_path} does not match '
                         f'the selected original antibody chains ({lengths})')
    if len(candidate_orders) != 1:
        raise ValueError('The order of the original antibody chains is ambiguous: '
                         f'candidate orders are {candidate_orders}')

    # These offsets are local to HADDOCK chain A; original-Universe resindices are
    # unsafe because preceding, unselected chains shift them.
    offset = 0
    for chain in candidate_orders[0]:
        chain_end = offset + len(sequences[chain])
        fused_antibody.residues[offset:chain_end].atoms.chainIDs = chain
        offset = chain_end

    if candidate_orders[0][-1] == 'B':
        # Keep a chain transition between antibody and antigen for pdb2gmx.
        antigen_chain = next(chain for chain in 'CDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                             if chain not in chain_ids)
        antigen.chainIDs = antigen_chain

    os.makedirs(os.path.dirname(os.path.abspath(output_pdb_path)), exist_ok=True)
    with mda.Writer(output_pdb_path, docked_u.atoms.n_atoms, reindex=False) as writer:
        writer.write(docked_u.atoms)
    return candidate_orders[0]


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

def pull_group_com(atoms, box):
    """Return the COM using GROMACS's default pull-group PBC reference atom.

    With pull-groupN-pbcatom=0, GROMACS uses the middle atom in index order
    (the lower middle for an even-sized group). Place each atom in its nearest
    periodic image around that reference *before* averaging with the masses.
    This also handles pull groups containing separately wrapped molecules.

    ``atoms`` must have the same order and masses as the pull group; ``box`` is
    the MDAnalysis unit cell in angstrom/degrees. No extra pull weights or custom
    PBC reference atoms are supported. The input coordinates are not modified.
    """
    if not len(atoms):
        raise ValueError('Cannot calculate the COM of an empty pull group')
    reference = atoms.positions[(len(atoms) - 1) // 2]
    positions = reference + minimize_vectors(atoms.positions - reference, box)
    return np.average(positions, axis=0, weights=atoms.masses)
