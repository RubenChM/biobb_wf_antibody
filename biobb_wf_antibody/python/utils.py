#!/usr/bin/env python3

"""Shared helpers of the antibody-antigen workflow.

The structures to dock are named by the 'reference', 'antibody' and 'antigen'
global properties of the configuration file, and every subworkflow needs the same
reading of them, so the parsing of those identifiers lives here instead of in one
of them. Which complex a run docks is decided before the workflow starts, by
array/launch_wf.py, which writes the three identifiers into the configuration file
of the run.
"""

import re


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
            raise ValueError("The '%s' global property is not set, it must be a PDB code "
                             "followed by the chains of interest, as in '4G6K_HL'" % key)
        identifiers[key] = str(identifier).strip()

    ref_code, ref_antibody_chains, ref_antigen_chains, ref_model = parse_identifier(identifiers['reference'])
    if not ref_antigen_chains:
        raise ValueError("The 'reference' identifier '%s' does not declare the antigen "
                         "chains of the complex after a colon, as in '4G6M_HL:A'"
                         % identifiers['reference'])

    antibody_code, antibody_chains, _, antibody_model = parse_identifier(identifiers['antibody'])
    antigen_code, antigen_chains, _, antigen_model = parse_identifier(identifiers['antigen'])
    for key, chains, example in [('reference', ref_antibody_chains, '4G6M_HL:A'),
                                 ('antibody', antibody_chains, '4G6K_HL'),
                                 ('antigen', antigen_chains, '4I1B_A')]:
        if not chains:
            raise ValueError("The '%s' identifier '%s' does not declare any chain after "
                             "its PDB code, as in '%s'" % (key, identifiers[key], example))

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
