#!/usr/bin/env python3

"""Subworkflow 3: AWH-MD of the docked complex -> CDR-loop ensemble -> HADDOCK3.

The best model of the baseline docking is simulated with the antigen in place, so the
CDR loops are sampled in the environment that shapes them, and the Adaptive Weighted
Histogram (AWH) method biases the antibody-antigen centre of mass distance to push the
complex through the whole range of the association. Several walkers share one bias, the
conformations their concatenated trajectory samples are clustered, and the cluster
representatives are docked as an ensemble exactly as the free-MD ones of 2_MD.py, so
the three docking runs are directly comparable.

Every path and property comes from the step3_* sections of the configuration file.
The antibody-antigen complex to simulate, the restraints and the antigen come from
the 'dependency/step1_*' paths, and the IMGT-renumbered antibody the CDR regions are
mapped through from 'dependency/step2_19_anarcii', so the MD subworkflow has to have
run at least up to that step before this one.
"""

import argparse
import importlib
import os
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
from biobb_gromacs.gromacs.mdrun_multidir import mdrun_multidir
from biobb_gromacs.gromacs.pdb2gmx import pdb2gmx
from biobb_gromacs.gromacs.solvate import solvate
from biobb_gromacs.gromacs.trjcat import trjcat
from biobb_model.model.fix_side_chain import fix_side_chain

from utils import (cdr_ndx_selection, ensure_force_field, haddock_best_model, import_cdr,
                   read_interface, report_execution)

# The clustering-to-docking stages are identical to the ones of the free MD run, so they
# are reused instead of being written again. The subworkflow modules are named after
# their index, so they cannot be pulled in with a plain import statement
_md = importlib.import_module('2_MD')
build_docking_ensemble = _md.build_docking_ensemble
check_ndx_groups = _md.check_ndx_groups
clean_cluster_representatives = _md.clean_cluster_representatives
stage_docking_input = _md.stage_docking_input


def awh_minimum_distance(minimum_distance):
    """Lower edge of the AWH interval for a given interface distance, in angstrom.

    The interval has to start a bit closer than the closest interface contact of the
    docked complex, but not so close that the two sides clash: 1.5 A is a reasonable
    lower bound for a non-bonded contact, and an interface that is already tighter
    than that is not pushed any further.
    """
    abs_min = min(1.5, minimum_distance)
    return max(abs_min, minimum_distance * 0.9)


def awh_interval(global_log, complex_pdb_path, equilibrated_gro_path,
                 paratope, epitope, upper_margin):
    """Edges of the AWH sampling interval, in nm.

    The AWH coordinate is the distance between the centres of mass of the two chains,
    so the interval is centred on the value that distance has in the equilibrated
    complex: it is opened by as much as the interface would have to give to reach
    'awh_minimum_distance', and closed 'upper_margin' angstrom further out.

    The chains are identified on the docked complex, which still carries its chain
    identifiers, and their residues are then looked up in the equilibrated structure,
    which does not. Every distance is taken under the minimum image convention: the
    .gro written by mdrun wraps whole molecules into the box, so the antigen can sit
    one box vector away from the antibody even though the complex is intact, and the
    GROMACS pull code evaluates the coordinate the same way.
    """
    pdb_chains = mda.Universe(complex_pdb_path).select_atoms('protein')
    chain_a_residx = pdb_chains.select_atoms('chainID A').residues.resindices
    chain_b_residx = pdb_chains.select_atoms('chainID B').residues.resindices

    universe = mda.Universe(equilibrated_gro_path).select_atoms('protein')
    chain_a = universe.residues[chain_a_residx].atoms
    chain_b = universe.residues[chain_b_residx].atoms
    box = universe.dimensions

    # TODO: derive the interface by distance instead of reusing the paratope and the
    # epitope of the docking. Taking centres of mass, the protein can compress or
    # expand and they can end up closer or further away than intended.
    paratope_atoms = chain_a.select_atoms(f'resid {" ".join(map(str, paratope))}')
    epitope_atoms = chain_b.select_atoms(f'resid {" ".join(map(str, epitope))}')
    if not paratope_atoms or not epitope_atoms:
        raise ValueError('The paratope or the epitope did not resolve onto the '
                         f'equilibrated structure {equilibrated_gro_path}')
    minimum_distance = mda.lib.distances.distance_array(
        paratope_atoms.positions, epitope_atoms.positions, box=box).min()

    com_vector = mda.lib.distances.minimize_vectors(
        chain_a.center_of_mass() - chain_b.center_of_mass(), box=box)
    com_distance = np.linalg.norm(com_vector)

    awh_min = com_distance - minimum_distance + awh_minimum_distance(minimum_distance)
    awh_max = com_distance + upper_margin
    global_log.info(f'  Closest paratope-epitope contact: {minimum_distance:.2f} A')
    global_log.info(f'  Chain A - chain B centre of mass distance: {com_distance:.2f} A')
    global_log.info(f'  AWH sampling interval: {awh_min:.2f} - {awh_max:.2f} A')
    # The mdp file takes nanometres
    return awh_min / 10, awh_max / 10


def write_awh_mdp(template_path, output_mdp_path, awh_min, awh_max):
    """Fill the edges of the sampling interval into the AWH mdp template"""
    fu.create_dir(os.path.dirname(output_mdp_path))
    with open(template_path) as f:
        template = f.read()
    with open(output_mdp_path, 'w') as f:
        f.write(template.format(awh_min=awh_min, awh_max=awh_max))
    return output_mdp_path


def walker_tpr_paths(output_tpr_path, n_walkers):
    """One tpr path per walker, in the subdirectory mdrun -multidir expects.

    mdrun -multidir runs one walker per subdirectory of the multi-simulation folder,
    each of them holding a tpr file of the same name.
    """
    walkers_dir = os.path.dirname(output_tpr_path)
    tpr_name = os.path.basename(output_tpr_path)
    return [os.path.join(walkers_dir, f'walker_{i}', tpr_name) for i in range(n_walkers)]


def zip_walker_trajectories(walkers_output_dir, n_walkers, zip_path, traj_name='traj.trr'):
    """Join the trajectories of all the walkers in the ZIP file trjcat expects"""
    fu.create_dir(os.path.dirname(zip_path))
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for i in range(n_walkers):
            walker_traj = os.path.join(walkers_output_dir, f'walker_{i}', traj_name)
            if not os.path.isfile(walker_traj):
                raise FileNotFoundError(f'Walker {i} did not write {walker_traj}')
            zipf.write(walker_traj, arcname=f'traj_{i}.trr')
    return zip_path


def awh_workflow(global_log, global_prop, global_paths):
    """Subworkflow 3: AWH-MD of the complex and docking of its CDR-loop clusters"""
    cdr = import_cdr()

    global_log.info('step3_0_fix_side_chain: Model the missing side chains of the docked complex')
    paths = dict(global_paths['step3_0_fix_side_chain'])
    # The complex to simulate is the best model of the baseline docking, whose stage
    # number inside the HADDOCK3 run directory is not known in advance. HADDOCK3 writes
    # it gzipped, so it is decompressed into the step directory
    best_model = haddock_best_model(paths.pop('input_haddock_wf_data'),
                                    paths.pop('output_best_model_path'))
    global_log.info(f'  Best model of the baseline docking: {best_model}')
    fix_side_chain(input_pdb_path=best_model, properties=global_prop['step3_0_fix_side_chain'],
                   **paths)
    fixed_pdb = paths['output_pdb_path']

    global_log.info('step3_1_charmm36: Force field of the GROMACS steps')
    paths = global_paths['step3_1_charmm36']
    prop = global_prop['step3_1_charmm36']
    # The same directory the MD subworkflow downloaded it into, nothing is fetched
    # again when it is already there
    gmx_lib = ensure_force_field(paths['output_ff_path'], prop['url'], prop['force_field'])
    global_log.info(f'  {prop["force_field"]}.ff under {gmx_lib}')

    global_log.info('step3_2_pdb2gmx: Build the GROMACS topology of the complex')
    paths = global_paths['step3_2_pdb2gmx']
    pdb2gmx(**paths, properties=dict(global_prop['step3_2_pdb2gmx'], gmx_lib=gmx_lib))

    global_log.info('step3_3_editconf: Define the unit cell')
    paths = global_paths['step3_3_editconf']
    editconf(**paths, properties=global_prop['step3_3_editconf'])

    global_log.info('step3_4_solvate: Fill the unit cell with water molecules')
    paths = global_paths['step3_4_solvate']
    solvate(**paths, properties=global_prop['step3_4_solvate'])

    global_log.info('step3_5_grompp_ions: Portable binary run file for the ion generation')
    paths = global_paths['step3_5_grompp_ions']
    grompp(**paths, properties=dict(global_prop['step3_5_grompp_ions'], gmx_lib=gmx_lib))

    global_log.info('step3_6_genion: Neutralize the system')
    paths = global_paths['step3_6_genion']
    genion(**paths, properties=global_prop['step3_6_genion'])

    global_log.info('step3_7_grompp_min: Portable binary run file for the minimization')
    paths = global_paths['step3_7_grompp_min']
    grompp(**paths, properties=dict(global_prop['step3_7_grompp_min'], gmx_lib=gmx_lib))
    min_tpr = paths['output_tpr_path']

    global_log.info('step3_8_mdrun_min: Energetically minimize the system')
    paths = global_paths['step3_8_mdrun_min']
    mdrun(**paths, properties=global_prop['step3_8_mdrun_min'])

    global_log.info('step3_9_gmx_energy_min: Potential energy along the minimization')
    paths = global_paths['step3_9_gmx_energy_min']
    gmx_energy(**paths, properties=global_prop['step3_9_gmx_energy_min'])

    global_log.info('step3_10_grompp_nvt: Portable binary run file for the NVT equilibration')
    paths = global_paths['step3_10_grompp_nvt']
    grompp(**paths, properties=dict(global_prop['step3_10_grompp_nvt'], gmx_lib=gmx_lib))

    global_log.info('step3_11_mdrun_nvt: Equilibrate the system (NVT)')
    paths = global_paths['step3_11_mdrun_nvt']
    mdrun(**paths, properties=global_prop['step3_11_mdrun_nvt'])

    global_log.info('step3_12_gmx_energy_nvt: Temperature along the NVT equilibration')
    paths = global_paths['step3_12_gmx_energy_nvt']
    gmx_energy(**paths, properties=global_prop['step3_12_gmx_energy_nvt'])

    global_log.info('step3_13_grompp_npt: Portable binary run file for the NPT equilibration')
    paths = global_paths['step3_13_grompp_npt']
    grompp(**paths, properties=dict(global_prop['step3_13_grompp_npt'], gmx_lib=gmx_lib))

    global_log.info('step3_14_mdrun_npt: Equilibrate the system (NPT)')
    paths = global_paths['step3_14_mdrun_npt']
    mdrun(**paths, properties=global_prop['step3_14_mdrun_npt'])
    npt_gro = paths['output_gro_path']

    global_log.info('step3_15_gmx_energy_npt: Pressure and density along the NPT equilibration')
    paths = global_paths['step3_15_gmx_energy_npt']
    gmx_energy(**paths, properties=global_prop['step3_15_gmx_energy_npt'])

    global_log.info('step3_16_awh_interval: AWH sampling interval of the equilibrated complex')
    paths = global_paths['step3_16_awh_interval']
    prop = global_prop['step3_16_awh_interval']
    # The paratope and the epitope are the ones the docking restraints were built from
    interface = read_interface(paths['input_interface_txt_path'])
    awh_min, awh_max = awh_interval(global_log, fixed_pdb, npt_gro,
                                    interface['A'], interface['B'],
                                    prop['upper_margin'])

    global_log.info('step3_17_awh_mdp: mdp file of the multiple walkers AWH run')
    paths = global_paths['step3_17_awh_mdp']
    awh_mdp = write_awh_mdp(paths['input_mdp_template_path'], paths['output_mdp_path'],
                            awh_min, awh_max)

    global_log.info('step3_18_make_ndx_chains: Index file with the two pull groups')
    # The pull groups of the AWH coordinate, addressed by residue index: the .gro and
    # the tpr carry no chain identifiers, so the two chains are located on the docked
    # complex and their ordinals are what goes into the index file
    pdb_chains = mda.Universe(fixed_pdb).select_atoms('protein')
    chain_ranges = []
    for chain in ('A', 'B'):
        resindices = pdb_chains.select_atoms(f'chainID {chain}').residues.resindices
        if not len(resindices):
            raise ValueError(f'The docked complex {fixed_pdb} has no chain {chain}')
        chain_ranges.append(f'{resindices[0] + 1}-{resindices[-1] + 1}')
    paths = global_paths['step3_18_make_ndx_chains']
    prop = global_prop['step3_18_make_ndx_chains']
    prop['selection'] = (f'ri {chain_ranges[0]}\nname 17 chA\n'
                         f'ri {chain_ranges[1]}\nname 18 chB\nq')
    global_log.info(f'  chA: residues {chain_ranges[0]}, chB: residues {chain_ranges[1]}')
    make_ndx(**paths, properties=prop)
    chains_ndx = paths['output_ndx_path']

    global_log.info('step3_19_grompp_awh: Portable binary run file of every walker')
    paths = dict(global_paths['step3_19_grompp_awh'])
    prop = dict(global_prop['step3_19_grompp_awh'], gmx_lib=gmx_lib)
    n_walkers = prop.pop('n_walkers')
    walker_tprs = walker_tpr_paths(paths.pop('output_tpr_path'), n_walkers)
    # One identical tpr per walker: they only differ in the random state mdrun gives
    # them, the bias itself is shared through 'awh-share-multisim'
    for i, walker_tpr in enumerate(walker_tprs):
        global_log.info(f'  walker {i}: {walker_tpr}')
        fu.create_dir(os.path.dirname(walker_tpr))
        grompp(input_mdp_path=awh_mdp, input_ndx_path=chains_ndx,
               output_tpr_path=walker_tpr, properties=prop, **paths)

    global_log.info(f'step3_20_mdrun_multidir: Run the {n_walkers} AWH walkers')
    paths = global_paths['step3_20_mdrun_multidir']
    mdrun_multidir(input_tpr_path=walker_tprs[-1],
                   input_multifolder=os.path.dirname(os.path.dirname(walker_tprs[0])),
                   properties=global_prop['step3_20_mdrun_multidir'], **paths)
    walkers_output = paths['output_multifolder']

    global_log.info('step3_21_trjcat: Concatenate the trajectories of all the walkers')
    paths = dict(global_paths['step3_21_trjcat'])
    zip_walker_trajectories(walkers_output, n_walkers, paths['input_trj_zip_path'])
    trjcat(**paths, properties=global_prop['step3_21_trjcat'])
    trjcat_path = paths['output_trj_path']

    global_log.info('step3_22_gmx_image_cluster: Strip the solvent and repair the periodicity')
    # This system needs two things at once from the PBC treatment, and only 'cluster'
    # gives both: the antibody has to be whole, and the antigen has to sit in the same
    # periodic image as the antibody. 'res' leaves the antibody split across the
    # boundary in most frames of the concatenated trajectory, which made the CDR-loop
    # RMSD read tens of angstrom and produced spurious clusters, while 'whole' and
    # 'mol' repair molecules one at a time and leave the antigen looking detached even
    # though its minimum image distance shows the complex never came apart. 'cluster'
    # wraps all the atoms of the group around an iteratively updated centre of mass,
    # which pulls the antigen back into contact while keeping the antibody whole.
    # 'center' must stay False: gmx_image only takes the cluster branch when centring
    # is off, and 'cluster_selection' is then the group fed to trjconv.
    #
    # Every step that needs the topology of the AWH run takes the tpr of one of the
    # walkers, they are all the same system
    awh_tpr = walker_tprs[-1]
    paths = global_paths['step3_22_gmx_image_cluster']
    gmx_image(input_top_path=awh_tpr,
              properties=global_prop['step3_22_gmx_image_cluster'], **paths)
    imaged_traj = paths['output_traj_path']

    global_log.info('step3_23_gmx_trjconv_str_dry: Dry structure, the topology of the analysis')
    paths = global_paths['step3_23_gmx_trjconv_str_dry']
    gmx_trjconv_str(input_top_path=awh_tpr,
                    properties=global_prop['step3_23_gmx_trjconv_str_dry'], **paths)
    dry_gro = paths['output_str_path']

    global_log.info('step3_24_gmx_image_rot: Superpose the trajectory on the whole complex')
    paths = global_paths['step3_24_gmx_image_rot']
    gmx_image(**paths, properties=global_prop['step3_24_gmx_image_rot'])
    imaged_traj_rot = paths['output_traj_path']

    global_log.info('step3_25_make_ndx: Index file with the CDR and the framework regions')
    # The regions have to be located again for this system: the index file of the free
    # MD run holds absolute atom numbers of the antibody alone, whereas this one is the
    # complex, its antibody chains are merged in a different order and the antigen
    # reuses their residue numbers, so reusing it addresses the wrong atoms. The IMGT
    # ranges are therefore mapped by residue index through the renumbered antibody of
    # the MD subworkflow, matching the chains by sequence
    paths = global_paths['step3_25_make_ndx']
    anarcii_pdb = paths['input_anarcii_pdb_path']
    cdr_ri, chain_order, n_antibody_res = cdr.imgt_region_ri(dry_gro, anarcii_pdb, cdr.CDR_RANGES)
    fr_ri, _, _ = cdr.imgt_region_ri(dry_gro, anarcii_pdb, cdr.FRAMEWORK_RANGES)
    n_fv = len(cdr_ri) + len(fr_ri)
    global_log.info(f'  Antibody chain order in the AWH system: {", ".join(chain_order)} '
                    f'({n_antibody_res} residues)')
    global_log.info(f'  {len(cdr_ri)} CDR + {len(fr_ri)} framework = {n_fv} variable-domain '
                    f'residues, {n_antibody_res - n_fv} constant-domain residues excluded')
    # The ranges cover the V domain only, so they do not add up to the whole antibody:
    # what must hold is that they are disjoint and both land inside it
    if set(cdr_ri) & set(fr_ri):
        raise ValueError('The CDR and the framework regions overlap')
    if not cdr_ri or not fr_ri:
        raise ValueError('One of the regions came out empty')
    if max(cdr_ri + fr_ri) > n_antibody_res:
        raise ValueError('A region falls outside the antibody')
    prop = global_prop['step3_25_make_ndx']
    # The extra 'Antibody' group is what the clustering writes out: gmx_cluster writes
    # whole groups, and the antigen has no business being in the ensemble that is
    # docked against it further down
    prop['selection'] = cdr_ndx_selection(cdr_ri, fr_ri, cdr.ri_selection,
                                          antibody_res=n_antibody_res)
    make_ndx(input_structure_path=dry_gro, output_ndx_path=paths['output_ndx_path'],
             properties=prop)
    loop_ndx = paths['output_ndx_path']
    groups = check_ndx_groups(global_log, dry_gro, loop_ndx, cdr.read_ndx,
                              {'Loop_CA': len(cdr_ri), 'Framework_CA': len(fr_ri)})
    universe = mda.Universe(str(dry_gro))
    antibody = universe.atoms[np.array(groups['Antibody']) - 1]
    global_log.info(f'  [Antibody] {antibody.n_atoms} atoms / {antibody.n_residues} residues '
                    f'({universe.atoms.n_residues - antibody.n_residues} antigen residues '
                    'left out)')
    if antibody.n_residues != n_antibody_res:
        raise ValueError('The Antibody group does not cover the antibody: '
                         f'{antibody.n_residues} residues instead of {n_antibody_res}')

    global_log.info('step3_26_gmx_image_framework: Superpose the trajectory on the Fv framework')
    # Same reason as in the free MD run: gmx cluster fits and measures on one group, so
    # the framework fit is done here and the clustering below runs with 'nofit'
    paths = global_paths['step3_26_gmx_image_framework']
    gmx_image(**paths, properties=global_prop['step3_26_gmx_image_framework'])

    global_log.info('step3_27_gmx_cluster: Cluster the CDR-loop conformations')
    paths = global_paths['step3_27_gmx_cluster']
    gmx_cluster(**paths, properties=global_prop['step3_27_gmx_cluster'])
    cluster_pdb = paths['output_pdb_path']

    global_log.info('step3_28_clean_clusters: Prepare the representatives for HADDOCK3')
    paths = global_paths['step3_28_clean_clusters']
    clean_cluster_representatives(cluster_pdb, paths['output_pdb_path'])
    clusters_clean = paths['output_pdb_path']

    global_log.info('step3_29_build_ensemble: Ensemble of the representatives and the '
                    'experimental structure')
    paths = global_paths['step3_29_build_ensemble']
    build_docking_ensemble(clusters_clean, paths['input_antibody_pdb_path'],
                           paths['output_zip_path'], paths['output_pdb_path'])
    ensemble_pdb = paths['output_pdb_path']

    global_log.info('step3_30_haddock3_run: Dock the CDR-loop ensemble against the antigen')
    paths = global_paths['step3_30_haddock3_run']
    stage_docking_input(paths, global_prop['step3_30_haddock3_run'], ensemble_pdb)

    return {
        'complex_pdb_path': fixed_pdb,
        'awh_interval_nm': (awh_min, awh_max),
        'awh_mdp_path': awh_mdp,
        'min_tpr_path': min_tpr,
        'walkers_output': walkers_output,
        'trjcat_path': trjcat_path,
        'dry_gro_path': dry_gro,
        'imaged_traj_path': imaged_traj,
        'imaged_traj_rot_path': imaged_traj_rot,
        'loop_ndx_path': loop_ndx,
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

    awh_workflow(global_log, global_prop, global_paths)
    report_execution(global_log, conf, config, start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AWH-MD of the antibody-antigen complex and "
                                                 "docking of its CDR-loop clusters with HADDOCK3")
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
