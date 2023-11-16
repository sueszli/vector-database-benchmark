import pytest
from stanza.models.constituency.parse_tree import Tree
from stanza.models.constituency import tree_reader
from stanza.tests import *
pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def test_leaf_preterminal():
    if False:
        while True:
            i = 10
    foo = Tree(label='foo')
    assert foo.is_leaf()
    assert not foo.is_preterminal()
    assert len(foo.children) == 0
    assert str(foo) == 'foo'
    bar = Tree(label='bar', children=foo)
    assert not bar.is_leaf()
    assert bar.is_preterminal()
    assert len(bar.children) == 1
    assert str(bar) == '(bar foo)'
    baz = Tree(label='baz', children=[bar])
    assert not baz.is_leaf()
    assert not baz.is_preterminal()
    assert len(baz.children) == 1
    assert str(baz) == '(baz (bar foo))'

def test_yield_preterminals():
    if False:
        print('Hello World!')
    text = '((S (VP (VB Unban)) (NP (NNP Mox) (NNP Opal))))'
    trees = tree_reader.read_trees(text)
    preterminals = list(trees[0].yield_preterminals())
    assert len(preterminals) == 3
    assert str(preterminals) == '[(VB Unban), (NNP Mox), (NNP Opal)]'

def test_depth():
    if False:
        i = 10
        return i + 15
    text = '(foo) ((S (VP (VB Unban)) (NP (NNP Mox) (NNP Opal))))'
    trees = tree_reader.read_trees(text)
    assert trees[0].depth() == 0
    assert trees[1].depth() == 4

def test_unique_labels():
    if False:
        while True:
            i = 10
    '\n    Test getting the unique labels from a tree\n\n    Assumes tree_reader works, which should be fine since it is tested elsewhere\n    '
    text = '((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?))) ((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))'
    trees = tree_reader.read_trees(text)
    labels = Tree.get_unique_constituent_labels(trees)
    expected = ['NP', 'PP', 'ROOT', 'SBARQ', 'SQ', 'VP', 'WHNP']
    assert labels == expected

def test_unique_tags():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test getting the unique tags from a tree\n    '
    text = '((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))'
    trees = tree_reader.read_trees(text)
    tags = Tree.get_unique_tags(trees)
    expected = ['.', 'DT', 'IN', 'NN', 'VBZ', 'WP']
    assert tags == expected

def test_unique_words():
    if False:
        print('Hello World!')
    '\n    Test getting the unique words from a tree\n    '
    text = '((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))'
    trees = tree_reader.read_trees(text)
    words = Tree.get_unique_words(trees)
    expected = ['?', 'Who', 'in', 'seat', 'sits', 'this']
    assert words == expected

def test_rare_words():
    if False:
        while True:
            i = 10
    '\n    Test getting the unique words from a tree\n    '
    text = '((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))  ((SBARQ (NP (DT this) (NN seat)) (. ?)))'
    trees = tree_reader.read_trees(text)
    words = Tree.get_rare_words(trees, 0.5)
    expected = ['Who', 'in', 'sits']
    assert words == expected

def test_common_words():
    if False:
        while True:
            i = 10
    '\n    Test getting the unique words from a tree\n    '
    text = '((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))  ((SBARQ (NP (DT this) (NN seat)) (. ?)))'
    trees = tree_reader.read_trees(text)
    words = Tree.get_common_words(trees, 3)
    expected = ['?', 'seat', 'this']
    assert words == expected

def test_root_labels():
    if False:
        print('Hello World!')
    text = '( (SBARQ-FOO (WHNP-BAR (WP Who)) (SQ#ASDF (VP=1 (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))'
    trees = tree_reader.read_trees(text)
    assert ['ROOT'] == Tree.get_root_labels(trees)
    text = '( (SBARQ-FOO (WHNP-BAR (WP Who)) (SQ#ASDF (VP=1 (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))' + '( (SBARQ-FOO (WHNP-BAR (WP Who)) (SQ#ASDF (VP=1 (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))' + '( (SBARQ-FOO (WHNP-BAR (WP Who)) (SQ#ASDF (VP=1 (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))'
    trees = tree_reader.read_trees(text)
    assert ['ROOT'] == Tree.get_root_labels(trees)
    text = '(FOO) (BAR)'
    trees = tree_reader.read_trees(text)
    assert ['BAR', 'FOO'] == Tree.get_root_labels(trees)

def test_prune_none():
    if False:
        print('Hello World!')
    text = ['((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (-NONE- in) (NP (DT this) (NN seat))))) (. ?)))', '((SBARQ (WHNP (-NONE- Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))', '((SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (-NONE- this) (-NONE- seat))))) (. ?)))']
    expected = ['(ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (NP (DT this) (NN seat))))) (. ?)))', '(ROOT (SBARQ (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (NN seat))))) (. ?)))', '(ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in)))) (. ?)))']
    for (t, e) in zip(text, expected):
        trees = tree_reader.read_trees(t)
        assert len(trees) == 1
        tree = trees[0].prune_none()
        assert e == str(tree)

def test_simplify_labels():
    if False:
        while True:
            i = 10
    text = '( (SBARQ-FOO (WHNP-BAR (WP Who)) (SQ#ASDF (VP=1 (VBZ sits) (PP (IN in) (NP (DT this) (- -))))) (. ?)))'
    expected = '(ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in) (NP (DT this) (- -))))) (. ?)))'
    trees = tree_reader.read_trees(text)
    trees = [t.simplify_labels() for t in trees]
    assert len(trees) == 1
    assert expected == str(trees[0])

def test_remap_constituent_labels():
    if False:
        for i in range(10):
            print('nop')
    text = '(ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in)))) (. ?)))'
    expected = '(ROOT (FOO (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in)))) (. ?)))'
    label_map = {'SBARQ': 'FOO'}
    trees = tree_reader.read_trees(text)
    trees = [t.remap_constituent_labels(label_map) for t in trees]
    assert len(trees) == 1
    assert expected == str(trees[0])

def test_remap_constituent_words():
    if False:
        print('Hello World!')
    text = '(ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in)))) (. ?)))'
    expected = '(ROOT (SBARQ (WHNP (WP unban)) (SQ (VP (VBZ mox) (PP (IN opal)))) (. ?)))'
    word_map = {'Who': 'unban', 'sits': 'mox', 'in': 'opal'}
    trees = tree_reader.read_trees(text)
    trees = [t.remap_words(word_map) for t in trees]
    assert len(trees) == 1
    assert expected == str(trees[0])

def test_replace_words():
    if False:
        i = 10
        return i + 15
    text = '(ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in)))) (. ?)))'
    expected = '(ROOT (SBARQ (WHNP (WP unban)) (SQ (VP (VBZ mox) (PP (IN opal)))) (. ?)))'
    new_words = ['unban', 'mox', 'opal', '?']
    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    tree = trees[0]
    new_tree = tree.replace_words(new_words)
    assert expected == str(new_tree)

def test_compound_constituents():
    if False:
        return 10
    text = '((VP (VB Unban)))'
    trees = tree_reader.read_trees(text)
    assert Tree.get_compound_constituents(trees) == [('ROOT', 'VP')]
    text = '(ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in)))) (. ?)))'
    trees = tree_reader.read_trees(text)
    assert Tree.get_compound_constituents(trees) == [('PP',), ('ROOT', 'SBARQ'), ('SQ', 'VP'), ('WHNP',)]
    text = '((VP (VB Unban)))   (ROOT (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ sits) (PP (IN in)))) (. ?)))'
    trees = tree_reader.read_trees(text)
    assert Tree.get_compound_constituents(trees) == [('PP',), ('ROOT', 'SBARQ'), ('ROOT', 'VP'), ('SQ', 'VP'), ('WHNP',)]

def test_equals():
    if False:
        print('Hello World!')
    "\n    Check one tree from the actual dataset for ==\n\n    when built with compound Open, this didn't work because of a silly bug\n    "
    text = '(ROOT (S (NP (DT The) (NNP Arizona) (NNPS Corporations) (NNP Commission)) (VP (VBD authorized) (NP (NP (DT an) (ADJP (CD 11.5)) (NN %) (NN rate) (NN increase)) (PP (IN at) (NP (NNP Tucson) (NNP Electric) (NNP Power) (NNP Co.))) (, ,) (UCP (ADJP (ADJP (RB substantially) (JJR lower)) (SBAR (IN than) (S (VP (VBN recommended) (NP (JJ last) (NN month)) (PP (IN by) (NP (DT a) (NN commission) (NN hearing) (NN officer))))))) (CC and) (NP (NP (QP (RB barely) (PDT half)) (DT the) (NN rise)) (VP (VBN sought) (PP (IN by) (NP (DT the) (NN utility)))))))) (. .)))'
    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    tree = trees[0]
    assert tree == tree
    trees2 = tree_reader.read_trees(text)
    tree2 = trees2[0]
    assert tree is not tree2
    assert tree == tree2
CHINESE_LONG_LIST_TREE = '\n(ROOT\n (IP\n  (NP (NNP 证券法))\n  (VP\n   (PP\n    (IN 对)\n    (NP\n     (DNP\n      (NP\n       (NP (NNP 中国))\n       (NP\n        (NN 证券)\n        (NN 市场)))\n      (DEC 的))\n     (NP (NN 运作))))\n   (, ，)\n   (PP\n    (PP\n     (IN 从)\n     (NP\n      (NP (NN 股票))\n      (NP (VV 发行) (EC 、) (VV 交易))))\n    (, ，)\n    (PP\n     (VV 到)\n     (NP\n      (NP (NN 上市) (NN 公司) (NN 收购))\n      (EC 、)\n      (NP (NN 证券) (NN 交易所))\n      (EC 、)\n      (NP (NN 证券) (NN 公司))\n      (EC 、)\n      (NP (NN 登记) (NN 结算) (NN 机构))\n      (EC 、)\n      (NP (NN 交易) (NN 服务) (NN 机构))\n      (EC 、)\n      (NP (NN 证券业) (NN 协会))\n      (EC 、)\n      (NP (NN 证券) (NN 监督) (NN 管理) (NN 机构))\n      (CC 和)\n      (NP\n       (DNP\n        (NP (CP (CP (IP (VP (VV 违法))))))\n        (DEC 的))\n       (NP (NN 法律) (NN 责任))))))\n   (ADVP (RB 都))\n   (VP\n    (VV 作)\n    (AS 了)\n    (NP\n     (ADJP (JJ 详细))\n     (NP (NN 规定)))))\n  (. 。)))\n'
WEIRD_UNARY = '\n  (DNP\n    (NP (CP (CP (IP (VP (ASDF\n      (NP (NN 上市) (NN 公司) (NN 收购))\n      (EC 、)\n      (NP (NN 证券) (NN 交易所))\n      (EC 、)\n      (NP (NN 证券) (NN 公司))\n      (EC 、)\n      (NP (NN 登记) (NN 结算) (NN 机构))\n      (EC 、)\n      (NP (NN 交易) (NN 服务) (NN 机构))\n      (EC 、)\n      (NP (NN 证券业) (NN 协会))\n      (EC 、)\n      (NP (NN 证券) (NN 监督) (NN 管理) (NN 机构))))))))\n    (DEC 的))\n'

def test_count_unaries():
    if False:
        return 10
    trees = tree_reader.read_trees(CHINESE_LONG_LIST_TREE)
    assert len(trees) == 1
    assert trees[0].count_unary_depth() == 5
    trees = tree_reader.read_trees(WEIRD_UNARY)
    assert len(trees) == 1
    assert trees[0].count_unary_depth() == 5

def test_str_bracket_labels():
    if False:
        while True:
            i = 10
    text = '((S (VP (VB Unban)) (NP (NNP Mox) (NNP Opal))))'
    expected = '(_ROOT (_S (_VP (_VB Unban )_VB )_VP (_NP (_NNP Mox )_NNP (_NNP Opal )_NNP )_NP )_S )_ROOT'
    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    assert '{:L}'.format(trees[0]) == expected

def test_all_leaves_are_preterminals():
    if False:
        for i in range(10):
            print('nop')
    text = '((S (VP (VB Unban)) (NP (NNP Mox) (NNP Opal))))'
    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    assert trees[0].all_leaves_are_preterminals()
    text = '((S (VP (VB Unban)) (NP (Mox) (NNP Opal))))'
    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    assert not trees[0].all_leaves_are_preterminals()

def test_pretty_print():
    if False:
        i = 10
        return i + 15
    '\n    Pretty print a couple trees - newlines & indentation\n    '
    text = '(ROOT (S (VP (VB Unban)) (NP (NNP Mox) (NNP Opal)))) (ROOT (S (NP (DT The) (NNP Arizona) (NNPS Corporations) (NNP Commission)) (VP (VBD authorized) (NP (NP (DT an) (ADJP (CD 11.5)) (NN %) (NN rate) (NN increase)) (PP (IN at) (NP (NNP Tucson) (NNP Electric)))))))'
    trees = tree_reader.read_trees(text)
    assert len(trees) == 2
    expected = '(ROOT\n  (S\n    (VP (VB Unban))\n    (NP (NNP Mox) (NNP Opal))))\n'
    assert '{:P}'.format(trees[0]) == expected
    expected = '(ROOT\n  (S\n    (NP (DT The) (NNP Arizona) (NNPS Corporations) (NNP Commission))\n    (VP\n      (VBD authorized)\n      (NP\n        (NP\n          (DT an)\n          (ADJP (CD 11.5))\n          (NN %)\n          (NN rate)\n          (NN increase))\n        (PP\n          (IN at)\n          (NP (NNP Tucson) (NNP Electric)))))))\n'
    assert '{:P}'.format(trees[1]) == expected
    assert text == '{:O} {:O}'.format(*trees)

def test_reverse():
    if False:
        while True:
            i = 10
    text = "(ROOT (S (NP (PRP I)) (VP (VBP want) (S (VP (TO to) (VP (VB lick) (NP (NP (NNP Jennifer) (POS 's)) (NNS antennae))))))))"
    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    reversed_tree = trees[0].reverse()
    assert str(reversed_tree) == "(ROOT (S (VP (S (VP (VP (NP (NNS antennae) (NP (POS 's) (NNP Jennifer))) (VB lick)) (TO to))) (VBP want)) (NP (PRP I))))"