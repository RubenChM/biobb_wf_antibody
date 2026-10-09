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
