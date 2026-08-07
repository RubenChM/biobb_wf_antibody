from pathlib import Path
import yaml

# ============================================================================
# The complex to work with: an index into the 'complexes' list of workflow.yml
# ============================================================================

case_id = 14

WORKFLOW_CONFIG = Path(__file__).resolve().parent.parent / 'python' / 'workflow.yml'
with open(WORKFLOW_CONFIG) as _f:
    complexes = [tuple(ids) for ids in
                 yaml.safe_load(_f)['global_properties']['complexes']]
reference, antibody, antigen = complexes[case_id]

# ============================================================================
# Downloaded structures and the working directory of the case
# ============================================================================

data_path = Path('data/')
base_dir = data_path / '0_base'
out_path = data_path / f'case_{case_id}'


def entry_pdb_path(identifier):
    """Path of a downloaded entry, whose identifier starts with its PDB code"""
    return base_dir / f"{identifier.split('_')[0]}.pdb"


antibody_pdb_src = str(entry_pdb_path(antibody))
antigen_pdb_src = str(entry_pdb_path(antigen))
reference_pdb_src = str(entry_pdb_path(reference))

# ============================================================================
# HADDOCK3 baseline docking
# ============================================================================

prep_dir = out_path / '1_pre'
antibody_pdb_clean = str(prep_dir / 'antibody_clean.pdb')
antigen_pdb_clean = str(prep_dir / 'antigen_clean.pdb')
reference_pdb_antibody = str(prep_dir / 'reference_antibody.pdb')
reference_pdb_antigen = str(prep_dir / 'reference_antigen.pdb')
zip_file_path = str(prep_dir / 'reference.zip')  # the two halves, fed to pdb_merge
reference_pdb_clean = str(prep_dir / 'reference_clean.pdb')
reference_interface = str(prep_dir / 'reference_interface.txt')
antibody_actpass = str(prep_dir / 'antibody_actpass.txt')
antigen_actpass = str(prep_dir / 'antigen_actpass.txt')
ambig_tbl = str(prep_dir / 'ambig-paratope-epitope.tbl')
unambig_tbl = str(prep_dir / 'antibody-unambig.tbl')
dock_dir = out_path / '2_dock'
haddock_best_pdb = str(dock_dir / 'output' / 'run' / '10_seletopclusts' / 'cluster_1_model_1.pdb')

# ============================================================================
# Free MD of the antibody -> CDR loop clusters
# ============================================================================

MD_dir = out_path / '3_MD'
MD_antibody_pdb_chains = str(MD_dir / 'antibody_chains.pdb')
MD_fixed_pdb = str(MD_dir / 'antibody_fixed.pdb')
MD_output_pdb2gmx_gro = str(MD_dir / 'antibody_pdb2gmx.gro')
MD_output_pdb2gmx_top_zip = str(MD_dir / 'antibody_pdb2gmx_top.zip')
MD_output_editconf_gro = str(MD_dir / 'antibody_editconf.gro')
MD_output_solvate_gro = str(MD_dir / 'antibody_solvate.gro')
MD_output_solvate_top_zip = str(MD_dir / 'antibody_solvate_top.zip')
MD_output_gppion_tpr = str(MD_dir / 'antibody_gppion.tpr')
MD_output_genion_gro = str(MD_dir / 'antibody_genion.gro')
MD_output_genion_top_zip = str(MD_dir / 'antibody_genion_top.zip')
# The minimization and the equilibration mdp files are shared with the AWH section,
# hence no 'MD_' prefix, as with the 'input_mdp_awh*' ones below
input_mdp_min = MD_dir / "emin-charmm.mdp"
input_mdp_eq = MD_dir / "md_eq_posre_charmm36m.mdp"
input_mdp_md = MD_dir / "md_charmm36m.mdp"
MD_output_gppmin_tpr = MD_dir / 'antibody_gppmin.tpr'
MD_output_min_trr = MD_dir / 'antibody_min.trr'
MD_output_min_gro = MD_dir / 'antibody_min.gro'
MD_output_min_edr = MD_dir / 'antibody_min.edr'
MD_output_min_log = MD_dir / 'antibody_min.log'
MD_output_min_ene_xvg = MD_dir / 'antibody_min_ene.xvg'
MD_output_gppnpt_tpr = MD_dir / 'antibody_gppnpt.tpr'
MD_output_npt_trr = MD_dir / 'antibody_npt.trr'
MD_output_npt_gro = MD_dir / 'antibody_npt.gro'
MD_output_npt_edr = MD_dir / 'antibody_npt.edr'
MD_output_npt_log = MD_dir / 'antibody_npt.log'
MD_output_npt_cpt = MD_dir / 'antibody_npt.cpt'
MD_output_npt_pd_xvg = MD_dir / 'antibody_npt_PD.xvg'
MD_output_gppmd_tpr = MD_dir / 'antibody_gppmd.tpr'
MD_output_md_trr = MD_dir / 'antibody_md.trr'
MD_output_md_gro = MD_dir / 'antibody_md.gro'
MD_output_md_edr = MD_dir / 'antibody_md.edr'
MD_output_md_log = MD_dir / 'antibody_md.log'
MD_output_md_cpt = MD_dir / 'antibody_md.cpt'
MD_imaged_traj = MD_dir / 'antibody_imaged_traj.trr'
MD_dry_gro = MD_dir / 'antibody_md_dry.gro'
MD_imaged_traj_rot = MD_dir / 'antibody_imaged_traj_rot.trr'
MD_imaged_traj_fw = MD_dir / 'antibody_imaged_traj_fw.trr'
MD_anarcii_pdb = str(Path(MD_fixed_pdb).with_name('antibody_anarcii_imgt.pdb'))
MD_anarcii_gro = MD_dir / 'antibody_anarcii.gro'
MD_anarcii_zip = MD_dir / 'antibody_anarcii.zip'
MD_loop_ndx = MD_dir / 'antibody_loop.ndx'
MD_rms_exp = MD_dir / 'antibody_rms_exp.xvg'
MD_rms_fr = MD_dir / 'antibody_rms_fr.xvg'
MD_rms_loop = MD_dir / 'antibody_rms_loop.xvg'
MD_rmsf_fr = MD_dir / 'antibody_rmsf_fr.xvg'
MD_rmsf_loop = MD_dir / 'antibody_rmsf_loop.xvg'
MD_rgyr = MD_dir / 'antibody_rgyr.xvg'
MD_pdb_cluster = MD_dir / 'antibody_clusters.pdb'
MD_aligned_clusters_pdb = str(MD_dir / 'antibody_clusters_aligned.pdb')
MD_pdb_cluster_clean = str(MD_dir / 'antibody_clusters_clean.pdb')
MD_clusters_zip = str(MD_dir / 'antibody_clusters_clean.zip')
MD_antibody_dock_pdb = str(MD_dir / 'antibody_docking.pdb')
MD_dock_dir = MD_dir / 'docking'

# ============================================================================
# AWH-MD of the antibody-antigen complex -> CDR loop clusters
# ============================================================================

AWH_dir = out_path / '4_AWH'
AWH_fixed_pdb = str(AWH_dir / 'complex_fixed.pdb')
AWH_output_pdb2gmx_gro = str(AWH_dir / 'complex_pdb2gmx.gro')
AWH_output_pdb2gmx_top_zip = str(AWH_dir / 'complex_pdb2gmx_top.zip')
AWH_output_editconf_gro = str(AWH_dir / 'complex_editconf.gro')
AWH_output_solvate_gro = str(AWH_dir / 'complex_solvate.gro')
AWH_output_solvate_top_zip = str(AWH_dir / 'complex_solvate_top.zip')
AWH_output_gppion_tpr = str(AWH_dir / 'complex_gppion.tpr')
AWH_output_genion_gro = str(AWH_dir / 'complex_genion.gro')
AWH_output_genion_top_zip = str(AWH_dir / 'complex_genion_top.zip')
AWH_output_gppmin_tpr = AWH_dir / 'complex_gppmin.tpr'
AWH_output_min_trr = AWH_dir / 'complex_min.trr'
AWH_output_min_gro = AWH_dir / 'complex_min.gro'
AWH_output_min_edr = AWH_dir / 'complex_min.edr'
AWH_output_min_log = AWH_dir / 'complex_min.log'
AWH_output_min_ene_xvg = AWH_dir / 'complex_min_ene.xvg'
AWH_output_nvt_tpr = AWH_dir / 'complex_nvt.tpr'
AWH_output_nvt_trr = AWH_dir / 'complex_nvt.trr'
AWH_output_nvt_gro = AWH_dir / 'complex_nvt.gro'
AWH_output_nvt_edr = AWH_dir / 'complex_nvt.edr'
AWH_output_nvt_log = AWH_dir / 'complex_nvt.log'
AWH_output_nvt_cpt = AWH_dir / 'complex_nvt.cpt'
AWH_output_nvt_temp_xvg = AWH_dir / 'complex_nvt_temp.xvg'
AWH_output_npt_tpr = AWH_dir / 'complex_npt.tpr'
AWH_output_npt_trr = AWH_dir / 'complex_npt.trr'
AWH_output_npt_gro = AWH_dir / 'complex_npt.gro'
AWH_output_npt_edr = AWH_dir / 'complex_npt.edr'
AWH_output_npt_log = AWH_dir / 'complex_npt.log'
AWH_output_npt_cpt = AWH_dir / 'complex_npt.cpt'
AWH_output_npt_pd_xvg = AWH_dir / 'complex_npt_PD.xvg'
AWH_input_ndx = str(AWH_dir / "index2.ndx")
input_mdp_awh_mult = AWH_dir / "awh_md_mult.mdp"
n_walkers = 4
walker_dir = AWH_dir / 'walkers' / f"walker_{n_walkers - 1}"
AWH_output_gppmd_mult_tpr = walker_dir / 'complex_gppmd_mult.tpr'
input_trj_zip_path = AWH_dir / 'walkers_traj_cat.zip'
AWH_trjcat_path = AWH_dir / 'complex_md_awh_cat.xtc'
AWH_imaged_traj = AWH_dir / 'complex_imaged_traj.trr'
AWH_dry_gro = AWH_dir / 'complex_md_dry.gro'
AWH_imaged_traj_rot = AWH_dir / 'complex_imaged_traj_rot.trr'
AWH_loop_ndx = AWH_dir / 'complex_loop.ndx'
AWH_imaged_traj_fw = AWH_dir / 'complex_imaged_traj_fw.trr'
AWH_rms_backbone = AWH_dir / 'complex_rms_backbone.xvg'
AWH_rms_fr = AWH_dir / 'complex_rms_fr.xvg'
AWH_rms_loop = AWH_dir / 'complex_rms_loop.xvg'
AWH_rmsf_fr = AWH_dir / 'complex_rmsf_fr.xvg'
AWH_rmsf_loop = AWH_dir / 'complex_rmsf_loop.xvg'
AWH_pdb_cluster = AWH_dir / 'complex_clusters.pdb'
AWH_pdb_cluster_clean = str(AWH_dir / 'complex_clusters_clean.pdb')
AWH_clusters_zip = str(AWH_dir / 'complex_clusters_clean.zip')
AWH_antibody_dock_pdb = str(AWH_dir / 'antibody_docking.pdb')
AWH_dock_dir = AWH_dir / 'docking'
