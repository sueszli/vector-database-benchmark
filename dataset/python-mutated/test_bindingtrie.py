"""Tests for the BindingTrie."""
import string
import itertools
import textwrap
import pytest
from qutebrowser.qt.gui import QKeySequence
from qutebrowser.keyinput import basekeyparser
from qutebrowser.keyinput import keyutils
from unit.keyinput import test_keyutils

@pytest.mark.parametrize('entered, configured, match_type', test_keyutils.TestKeySequence.MATCH_TESTS)
def test_matches_single(entered, configured, match_type):
    if False:
        return 10
    entered = keyutils.KeySequence.parse(entered)
    configured = keyutils.KeySequence.parse(configured)
    trie = basekeyparser.BindingTrie()
    trie[configured] = 'eeloo'
    command = 'eeloo' if match_type == QKeySequence.SequenceMatch.ExactMatch else None
    result = basekeyparser.MatchResult(match_type=match_type, command=command, sequence=entered)
    assert trie.matches(entered) == result

def test_str():
    if False:
        i = 10
        return i + 15
    bindings = {keyutils.KeySequence.parse('a'): 'cmd-a', keyutils.KeySequence.parse('ba'): 'cmd-ba', keyutils.KeySequence.parse('bb'): 'cmd-bb', keyutils.KeySequence.parse('cax'): 'cmd-cax', keyutils.KeySequence.parse('cby'): 'cmd-cby'}
    trie = basekeyparser.BindingTrie()
    trie.update(bindings)
    expected = '\n        a:\n          => cmd-a\n\n        b:\n          a:\n            => cmd-ba\n          b:\n            => cmd-bb\n\n        c:\n          a:\n            x:\n              => cmd-cax\n          b:\n            y:\n              => cmd-cby\n    '
    assert str(trie) == textwrap.dedent(expected).lstrip('\n')

@pytest.mark.parametrize('configured, expected', [([], [('a', QKeySequence.SequenceMatch.NoMatch), ('', QKeySequence.SequenceMatch.NoMatch)]), (['abcd'], [('abcd', QKeySequence.SequenceMatch.ExactMatch), ('abc', QKeySequence.SequenceMatch.PartialMatch)]), (['aa', 'ab', 'ac', 'ad'], [('ac', QKeySequence.SequenceMatch.ExactMatch), ('a', QKeySequence.SequenceMatch.PartialMatch), ('f', QKeySequence.SequenceMatch.NoMatch), ('acd', QKeySequence.SequenceMatch.NoMatch)]), (['aaaaaaab', 'aaaaaaac', 'aaaaaaad'], [('aaaaaaab', QKeySequence.SequenceMatch.ExactMatch), ('z', QKeySequence.SequenceMatch.NoMatch)]), (string.ascii_letters, [('a', QKeySequence.SequenceMatch.ExactMatch), ('!', QKeySequence.SequenceMatch.NoMatch)])])
def test_matches_tree(configured, expected, benchmark):
    if False:
        i = 10
        return i + 15
    trie = basekeyparser.BindingTrie()
    trie.update({keyutils.KeySequence.parse(keys): 'eeloo' for keys in configured})

    def run():
        if False:
            while True:
                i = 10
        for (entered, match_type) in expected:
            sequence = keyutils.KeySequence.parse(entered)
            command = 'eeloo' if match_type == QKeySequence.SequenceMatch.ExactMatch else None
            result = basekeyparser.MatchResult(match_type=match_type, command=command, sequence=sequence)
            assert trie.matches(sequence) == result
    benchmark(run)

@pytest.mark.parametrize('configured', [['a'], itertools.permutations('asdfghjkl', 3)])
def test_bench_create(configured, benchmark):
    if False:
        print('Hello World!')
    bindings = {keyutils.KeySequence.parse(keys): 'dres' for keys in configured}

    def run():
        if False:
            i = 10
            return i + 15
        trie = basekeyparser.BindingTrie()
        trie.update(bindings)
    benchmark(run)