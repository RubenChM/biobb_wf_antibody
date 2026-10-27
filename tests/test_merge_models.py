"""merge_models gives the copies of an assembly distinct chains (no Modeller needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from antibody_wf.utils import merge_models, parse_seqres

ATOM = 'ATOM      1  N   ALA {}   1      {:>7.3f}  38.598  -3.744  1.00 51.64           N'
SEQRES = 'SEQRES   1 A    1  ALA'


def test_repeated_chain_gets_next_free_id_and_seqres():
    text = '\n'.join([SEQRES, 'MODEL        1', ATOM.format('A', 1), 'ENDMDL',
                      'MODEL        2', ATOM.format('A', 2), 'ENDMDL'])
    out = merge_models(text)
    assert [l[21] for l in out.splitlines() if l.startswith('ATOM')] == ['A', 'B']
    assert parse_seqres(out) == {'A': 'A', 'B': 'A'}
    assert 'MODEL' not in out


def test_distinct_chains_are_untouched():
    text = '\n'.join(['MODEL        1', ATOM.format('A', 1), ATOM.format('B', 2), 'ENDMDL'])
    assert [l[21] for l in merge_models(text).splitlines() if l.startswith('ATOM')] == ['A', 'B']


def test_copies_named_among_the_selected_chains():
    """The 4FQI assembly: three MODELs of the antigen A/B and the Fab H/L. Among the antigen
    chains alone the copies become C/D and E/F, as the benchmark names them; among every chain
    the Fab copies take C-F first."""
    def model(n):
        return [f'MODEL        {n}'] + [ATOM.format(c, n) for c in 'ABHL'] + ['ENDMDL']
    text = '\n'.join(model(1) + model(2) + model(3))

    def chains(out):
        return ''.join(l[21] for l in out.splitlines() if l.startswith('ATOM'))
    assert chains(merge_models(text, list('ABEFCD'))) == 'ABCDEF'
    assert chains(merge_models(text)) == 'ABHLCDEFGIJK'
