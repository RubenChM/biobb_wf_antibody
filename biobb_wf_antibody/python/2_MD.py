#!/usr/bin/env python3

"""Subworkflow 2: free MD of the unbound antibody -> CDR-loop ensemble -> HADDOCK3.

The antibody is simulated on its own, the conformations its CDR loops sample along
the trajectory are clustered, and the cluster representatives are docked against the
antigen as an ensemble, with the same restraints as the baseline run of 1_haddock.py.
Comparing that docking with the baseline is what says whether the loop conformations
sampled by the MD help.

Every path and property comes from the step2_* sections of the configuration file,
and everything reused from the docking run is picked up through 'dependency/step1_*'
paths, so this subworkflow also runs on its own on a working directory where
1_haddock.py has already run.
"""

import argparse
import os
import shutil
import time
import zipfile

import MDAnalysis as mda
import numpy as np
from biobb_common.configuration import settings
from biobb_common.tools import file_utils as fu

from biobb_analysis.gromacs.gmx_cluster import gmx_cluster
from biobb_analysis.gromacs.gmx_energy import gmx_energy
from biobb_analysis.gromacs.gmx_image import gmx_image
from biobb_analysis.gromacs.gmx_trjconv_str import gmx_trjconv_str
from biobb_gromacs.gromacs.editconf import editconf
from biobb_gromacs.gromacs.genion import genion
from biobb_gromacs.gromacs.grompp import grompp
from biobb_gromacs.gromacs.make_ndx import make_ndx
from biobb_gromacs.gromacs.mdrun import mdrun
from biobb_gromacs.gromacs.pdb2gmx import pdb2gmx
from biobb_gromacs.gromacs.solvate import solvate
from biobb_haddock.haddock.haddock3_run import haddock3_run
from biobb_haddock.utils.anarcii import anarcii
from biobb_model.model.fix_side_chain import fix_side_chain
from biobb_pdb_tools.pdb_tools import biobb_pdb_chain
from biobb_pdb_tools.pdb_tools import biobb_pdb_chainxseg
from biobb_pdb_tools.pdb_tools import biobb_pdb_reres
from biobb_pdb_tools.pdb_tools import biobb_pdb_selchain
from biobb_pdb_tools.pdb_tools import biobb_pdb_tidy
from biobb_pdb_tools.pdb_tools.biobb_pdb_mkensemble import biobb_pdb_mkensemble
from biobb_pdb_tools.pdb_tools.biobb_pdb_splitmodel import biobb_pdb_splitmodel

import cdr
from utils import (cdr_ndx_selection, ensure_force_field, pdb_tools_pipeline,
                   report_execution, resolve_complex)


def clean_cluster_representatives(input_pdb_path, output_pdb_path):
    """Relabel the cluster representatives to meet the HADDOCK3 requirements.

    The same cleaning the antibody of the baseline docking gets in 1_haddock.py,
    minus the per-chain part: the representatives written by gmx_cluster already
    hold a single body.
    """
    input_pdb_path = os.path.abspath(input_pdb_path)
    step_path = os.path.dirname(output_pdb_path)
    fu.create_dir(step_path)

    with fu.change_dir(step_path):
        steps = [
            (biobb_pdb_reres.biobb_pdb_reres, {'number': 1}),       # 1. Renumber the residues starting from 1
            (biobb_pdb_chain.biobb_pdb_chain, {'chain': 'A'}),      # 2. Modify the chain identifier column
            (biobb_pdb_chainxseg.biobb_pdb_chainxseg, {}),          # 3. Swap the segment identifier for the chain identifier
            (biobb_pdb_tidy.biobb_pdb_tidy, {'strict': True})       # 4. Adhere to the format specifications
        ]
        pdb_tools_pipeline(input_pdb_path, output_pdb_path, steps)


def build_docking_ensemble(cluster_pdb_path, experimental_pdb_path, zip_path, output_pdb_path):
    """Build the multi-model PDB file HADDOCK3 docks as an ensemble.

    It holds the cluster representatives plus the experimental structure, so the
    conformation the baseline run docked is always part of the ensemble. The
    experimental one is added last ('zz_' prefix, the ZIP entries are read in
    alphabetical order) to keep the numbering of the representatives.
    """
    fu.create_dir(os.path.dirname(output_pdb_path))
    # 1. Split the multi-model PDB of cluster representatives into one PDB per model
    biobb_pdb_splitmodel(input_file_path=cluster_pdb_path, output_file_path=zip_path)
    # 2. Add the cleaned experimental structure to the ZIP
    with zipfile.ZipFile(zip_path, 'a') as zipf:
        zipf.write(experimental_pdb_path,
                   arcname=f'zz_{os.path.basename(experimental_pdb_path)}')
    # 3. Build the multi-model (ensemble) PDB file
    biobb_pdb_mkensemble(input_file_path=zip_path, output_file_path=output_pdb_path)


def check_ndx_groups(global_log, structure_path, ndx_path, read_ndx, expected):
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


def stage_docking_input(paths, prop, antibody_pdb_path):
    """Copy the docking input files under the names haddock_config.cfg expects.

    The antigen, the reference complex and both restraint tables are the ones of
    the baseline docking, only the antibody differs: here it is the CDR-loop
    ensemble instead of the single experimental structure. The configuration file and
    the properties of the step have to hold the same protocol as the baseline run,
    which is what makes the three docking runs comparable.
    """
    fu.create_dir(paths['input_haddock_wf_data'])
    for name, src in [('antibody.pdb', antibody_pdb_path),
                      ('antigen.pdb', paths['input_antigen_pdb_path']),
                      ('reference.pdb', paths['input_reference_pdb_path']),
                      ('ambig.tbl', paths['input_ambig_tbl_path']),
                      ('unambig.tbl', paths['input_unambig_tbl_path'])]:
        shutil.copy2(src, os.path.join(paths['input_haddock_wf_data'], name))
    haddock3_run(haddock_config_path=paths['haddock_config_path'],
                 input_haddock_wf_data=paths['input_haddock_wf_data'],
                 output_haddock_wf_data=paths['output_haddock_wf_data'],
                 properties=prop)


def md_workflow(global_log, global_prop, global_paths, complex_ids=None):
    """Subworkflow 2: free MD of the antibody and docking of its CDR-loop clusters

    'complex_ids' is the already resolved selection when this subworkflow is run from
    workflow.py.
    """
    if complex_ids is None:
        complex_ids = resolve_complex(global_prop['step2_0_biobb_pdb_selchain'])

    global_log.info('step2_0_biobb_pdb_selchain: Extract the antibody chains '
                    f'{complex_ids["antibody"]["chains"]}')
    paths = global_paths['step2_0_biobb_pdb_selchain']
    prop = global_prop['step2_0_biobb_pdb_selchain']
    prop['chains'] = complex_ids['antibody']['chains']
    biobb_pdb_selchain.biobb_pdb_selchain(**paths, properties=prop)

    global_log.info('step2_1_fix_side_chain: Model the missing side chain atoms')
    paths = global_paths['step2_1_fix_side_chain']
    fix_side_chain(**paths, properties=global_prop['step2_1_fix_side_chain'])

    global_log.info('step2_2_charmm36: Force field of the GROMACS steps')
    paths = global_paths['step2_2_charmm36']
    prop = global_prop['step2_2_charmm36']
    # Every pdb2gmx and grompp step below is given this directory as its GMXLIB
    gmx_lib = ensure_force_field(paths['output_ff_path'], prop['url'], prop['force_field'])
    global_log.info(f'  {prop["force_field"]}.ff under {gmx_lib}')

    global_log.info('step2_3_pdb2gmx: Build the GROMACS topology of the antibody')
    paths = global_paths['step2_3_pdb2gmx']
    pdb2gmx(**paths, properties=dict(global_prop['step2_3_pdb2gmx'], gmx_lib=gmx_lib))

    global_log.info('step2_4_editconf: Define the unit cell')
    paths = global_paths['step2_4_editconf']
    editconf(**paths, properties=global_prop['step2_4_editconf'])

    global_log.info('step2_5_solvate: Fill the unit cell with water molecules')
    paths = global_paths['step2_5_solvate']
    solvate(**paths, properties=global_prop['step2_5_solvate'])

    global_log.info('step2_6_grompp_ions: Portable binary run file for the ion generation')
    paths = global_paths['step2_6_grompp_ions']
    grompp(**paths, properties=dict(global_prop['step2_6_grompp_ions'], gmx_lib=gmx_lib))

    global_log.info('step2_7_genion: Neutralize the system')
    paths = global_paths['step2_7_genion']
    genion(**paths, properties=global_prop['step2_7_genion'])

    global_log.info('step2_8_grompp_min: Portable binary run file for the minimization')
    paths = global_paths['step2_8_grompp_min']
    grompp(**paths, properties=dict(global_prop['step2_8_grompp_min'], gmx_lib=gmx_lib))
    min_tpr = paths['output_tpr_path']

    global_log.info('step2_9_mdrun_min: Energetically minimize the system')
    paths = global_paths['step2_9_mdrun_min']
    mdrun(**paths, properties=global_prop['step2_9_mdrun_min'])

    global_log.info('step2_10_gmx_energy_min: Potential energy along the minimization')
    paths = global_paths['step2_10_gmx_energy_min']
    gmx_energy(**paths, properties=global_prop['step2_10_gmx_energy_min'])

    global_log.info('step2_11_grompp_npt: Portable binary run file for the NPT equilibration')
    paths = global_paths['step2_11_grompp_npt']
    grompp(**paths, properties=dict(global_prop['step2_11_grompp_npt'], gmx_lib=gmx_lib))

    global_log.info('step2_12_mdrun_npt: Equilibrate the system with position restraints (NPT)')
    paths = global_paths['step2_12_mdrun_npt']
    mdrun(**paths, properties=global_prop['step2_12_mdrun_npt'])

    global_log.info('step2_13_gmx_energy_npt: Pressure and density along the NPT equilibration')
    paths = global_paths['step2_13_gmx_energy_npt']
    gmx_energy(**paths, properties=global_prop['step2_13_gmx_energy_npt'])

    global_log.info('step2_14_grompp_md: Portable binary run file for the free MD run')
    paths = global_paths['step2_14_grompp_md']
    grompp(**paths, properties=dict(global_prop['step2_14_grompp_md'], gmx_lib=gmx_lib))

    global_log.info('step2_15_mdrun_md: Run the free MD simulation of the antibody')
    paths = global_paths['step2_15_mdrun_md']
    mdrun(**paths, properties=global_prop['step2_15_mdrun_md'])

    global_log.info('step2_16_gmx_image_nojump: Strip the solvent and undo the periodic jumps')
    paths = global_paths['step2_16_gmx_image_nojump']
    gmx_image(**paths, properties=global_prop['step2_16_gmx_image_nojump'])
    imaged_traj = paths['output_traj_path']

    global_log.info('step2_17_gmx_trjconv_str_dry: Dry structure, the topology of the analysis')
    paths = global_paths['step2_17_gmx_trjconv_str_dry']
    gmx_trjconv_str(**paths, properties=global_prop['step2_17_gmx_trjconv_str_dry'])
    dry_gro = paths['output_str_path']

    global_log.info('step2_18_gmx_image_rot: Superpose the trajectory on the whole antibody')
    paths = global_paths['step2_18_gmx_image_rot']
    gmx_image(**paths, properties=global_prop['step2_18_gmx_image_rot'])
    imaged_traj_rot = paths['output_traj_path']

    global_log.info('step2_19_anarcii: Renumber the antibody with the IMGT unique numbering')
    paths = global_paths['step2_19_anarcii']
    anarcii(**paths, properties=global_prop['step2_19_anarcii'])
    anarcii_pdb = paths['output_pdb_path']

    global_log.info('step2_20_pdb2gmx_anarcii: GROMACS structure of the renumbered antibody')
    paths = global_paths['step2_20_pdb2gmx_anarcii']
    pdb2gmx(**paths, properties=dict(global_prop['step2_20_pdb2gmx_anarcii'], gmx_lib=gmx_lib))
    anarcii_gro = paths['output_gro_path']

    global_log.info('step2_21_make_ndx: Index file with the CDR and the framework regions')
    # CDR (IMGT 27-38, 56-65, 105-117) and framework (1-26, 39-55, 66-104, 118-129)
    # regions of the V domain, resolved onto this system. The regions are intersected
    # with the C-alpha group to build the Loop_CA / Framework_CA groups the framework
    # fit and the clustering below run on
    cdr_ri, _, _ = cdr.imgt_region_ri(anarcii_gro, anarcii_pdb, cdr.CDR_RANGES)
    fr_ri, _, _ = cdr.imgt_region_ri(anarcii_gro, anarcii_pdb, cdr.FRAMEWORK_RANGES)
    global_log.info(f'  {len(cdr_ri)} CDR + {len(fr_ri)} framework variable-domain residues')
    paths = global_paths['step2_21_make_ndx']
    prop = global_prop['step2_21_make_ndx']
    prop['selection'] = cdr_ndx_selection(cdr_ri, fr_ri, cdr.ri_selection)
    make_ndx(**paths, properties=prop)
    loop_ndx = paths['output_ndx_path']
    check_ndx_groups(global_log, dry_gro, loop_ndx, cdr.read_ndx,
                     {'Loop_CA': len(cdr_ri), 'Framework_CA': len(fr_ri)})

    global_log.info('step2_22_gmx_image_framework: Superpose the trajectory on the Fv framework')
    # gmx cluster uses a single group for both the least-squares fit and the RMSD, so
    # it cannot fit on the framework while measuring the loops. The fit is done here
    # and the clustering below runs with 'nofit': the framework is held fixed, so
    # nothing re-fits the loops onto themselves and hides how far they actually move
    paths = global_paths['step2_22_gmx_image_framework']
    gmx_image(**paths, properties=global_prop['step2_22_gmx_image_framework'])

    global_log.info('step2_23_gmx_cluster: Cluster the CDR-loop conformations')
    # Clustering on the C-alpha atoms of the loops, not on every atom of the CDR
    # residues: side-chain rotamers add a large offset that has nothing to do with
    # loop conformation, and HADDOCK refines side chains itself
    paths = global_paths['step2_23_gmx_cluster']
    gmx_cluster(**paths, properties=global_prop['step2_23_gmx_cluster'])
    cluster_pdb = paths['output_pdb_path']

    global_log.info('step2_24_clean_clusters: Prepare the representatives for HADDOCK3')
    paths = global_paths['step2_24_clean_clusters']
    clean_cluster_representatives(cluster_pdb, paths['output_pdb_path'])
    clusters_clean = paths['output_pdb_path']

    global_log.info('step2_25_build_ensemble: Ensemble of the representatives and the '
                    'experimental structure')
    paths = global_paths['step2_25_build_ensemble']
    build_docking_ensemble(clusters_clean, paths['input_antibody_pdb_path'],
                           paths['output_zip_path'], paths['output_pdb_path'])
    ensemble_pdb = paths['output_pdb_path']

    global_log.info('step2_26_haddock3_run: Dock the CDR-loop ensemble against the antigen')
    paths = global_paths['step2_26_haddock3_run']
    stage_docking_input(paths, global_prop['step2_26_haddock3_run'], ensemble_pdb)

    return {
        'anarcii_pdb_path': anarcii_pdb,
        'dry_gro_path': dry_gro,
        'imaged_traj_path': imaged_traj,
        'imaged_traj_rot_path': imaged_traj_rot,
        'loop_ndx_path': loop_ndx,
        'min_tpr_path': min_tpr,
        'force_field_path': gmx_lib,
        'cluster_pdb_path': cluster_pdb,
        'ensemble_pdb_path': ensemble_pdb,
        'haddock_wf_data': paths['output_haddock_wf_data'],
    }


def main(config):
    start_time = time.time()
    conf = settings.ConfReader(config)
    global_log, _ = fu.get_logs(path=conf.get_working_dir_path(), light_format=True)
    global_prop = conf.get_prop_dic(global_log=global_log)
    global_paths = conf.get_paths_dic()

    md_workflow(global_log, global_prop, global_paths)
    report_execution(global_log, conf, config, start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Free MD of the antibody and docking of "
                                                 "its CDR-loop clusters with HADDOCK3")
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
