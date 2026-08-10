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
the GROMACS subworkflows share, the CHARMM36 force field and the CDR/framework
definitions of the notebook.
"""

import glob
import gzip
import importlib.util
import os
import re
import shutil
import tarfile
import time
import urllib.request
import zipfile


def parse_identifier(identifier):
    """Split a structure identifier into its PDB code, its chains and its model.

    The chains of interest follow the PDB code, the ones after the colon are the
    antigen chains of a reference complex. A trailing '(<n>)' is the model to
    extract from the entry, needed by the NMR ensembles that hold one conformer per
    model:
      '4G6K_HL'     -> ('4G6K', 'H,L', None, None)
      '4G6M_HL:A'   -> ('4G6M', 'H,L', 'A', None)
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
    """Join several PDB files in a single ZIP file, as expected by pdb_merge"""
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for pdb_path in pdb_paths:
            zipf.write(pdb_path, arcname=os.path.basename(pdb_path))
    return zip_file_path


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

# The IMGT CDR/framework definitions and the mapping of those ranges onto a given
# system live in the notebooks folder, next to the notebook that documents them.
# They are loaded from there instead of being copied here, so the two stay in sync.
CDR_MODULE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'notebooks', 'cdr.py')


def import_cdr():
    """Import the notebooks/cdr.py module"""
    if not os.path.isfile(CDR_MODULE_PATH):
        raise FileNotFoundError(f'The CDR definitions are expected at {CDR_MODULE_PATH}')
    spec = importlib.util.spec_from_file_location('cdr', CDR_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
