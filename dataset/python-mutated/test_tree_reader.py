import pytest
from stanza.models.constituency import tree_reader
from stanza.models.constituency.tree_reader import MixedTreeError, UnclosedTreeError, UnlabeledTreeError
from stanza.tests import *
pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def test_simple():
    if False:
        print('Hello World!')
    '\n    Tests reading two simple trees from the same text\n    '
    text = '(VB Unban) (NNP Opal)'
    trees = tree_reader.read_trees(text)
    assert len(trees) == 2
    assert trees[0].is_preterminal()
    assert trees[0].label == 'VB'
    assert trees[0].children[0].label == 'Unban'
    assert trees[1].is_preterminal()
    assert trees[1].label == 'NNP'
    assert trees[1].children[0].label == 'Opal'

def test_newlines():
    if False:
        while True:
            i = 10
    '\n    The same test should work if there are newlines\n    '
    text = '(VB Unban)\n\n(NNP Opal)'
    trees = tree_reader.read_trees(text)
    assert len(trees) == 2

def test_parens():
    if False:
        i = 10
        return i + 15
    '\n    Parens should be escaped in the tree files and escaped when written\n    '
    text = '(-LRB- -LRB-) (-RRB- -RRB-)'
    trees = tree_reader.read_trees(text)
    assert len(trees) == 2
    assert trees[0].label == '-LRB-'
    assert trees[0].children[0].label == '('
    assert '{}'.format(trees[0]) == '(-LRB- -LRB-)'
    assert trees[1].label == '-RRB-'
    assert trees[1].children[0].label == ')'
    assert '{}'.format(trees[1]) == '(-RRB- -RRB-)'

def test_complicated():
    if False:
        print('Hello World!')
    '\n    A more complicated tree that should successfully read\n    '
    text = '( (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))'
    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    tree = trees[0]
    assert not tree.is_leaf()
    assert not tree.is_preterminal()
    assert tree.label == 'ROOT'
    assert len(tree.children) == 1
    assert tree.children[0].label == 'SBARQ'
    assert len(tree.children[0].children) == 3
    assert [x.label for x in tree.children[0].children] == ['WHNP', 'SQ', '.']

def test_one_word():
    if False:
        print('Hello World!')
    '\n    Check that one node trees are correctly read\n\n    probably not super relevant for the parsing use case\n    '
    text = '(FOO) (BAR)'
    trees = tree_reader.read_trees(text)
    assert len(trees) == 2
    assert trees[0].is_leaf()
    assert trees[0].label == 'FOO'
    assert trees[1].is_leaf()
    assert trees[1].label == 'BAR'

def test_missing_close_parens():
    if False:
        i = 10
        return i + 15
    '\n    Test the unclosed error condition\n    '
    text = '(Foo) \n (Bar \n zzz'
    try:
        trees = tree_reader.read_trees(text)
        raise AssertionError('Expected an exception')
    except UnclosedTreeError as e:
        assert e.line_num == 1

def test_mixed_tree():
    if False:
        i = 10
        return i + 15
    '\n    Test the mixed error condition\n    '
    text = '(Foo) \n (Bar) \n (Unban (Mox) Opal)'
    try:
        trees = tree_reader.read_trees(text)
        raise AssertionError('Expected an exception')
    except MixedTreeError as e:
        assert e.line_num == 2
    trees = tree_reader.read_trees(text, broken_ok=True)
    assert len(trees) == 3

def test_unlabeled_tree():
    if False:
        while True:
            i = 10
    '\n    Test the unlabeled error condition\n    '
    text = '(ROOT ((Foo) (Bar)))'
    try:
        trees = tree_reader.read_trees(text)
        raise AssertionError('Expected an exception')
    except UnlabeledTreeError as e:
        assert e.line_num == 0
    trees = tree_reader.read_trees(text, broken_ok=True)
    assert len(trees) == 1