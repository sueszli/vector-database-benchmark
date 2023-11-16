"""
Test the parser eval interface
"""
import pytest
import stanza
from stanza.models.constituency import tree_reader
from stanza.protobuf import EvaluateParserRequest, EvaluateParserResponse
from stanza.server.parser_eval import build_request, collate, EvaluateParser, ParseResult
from stanza.tests.server.test_java_protobuf_requests import check_tree
from stanza.tests import *
pytestmark = [pytest.mark.travis, pytest.mark.client]

def build_one_tree_treebank(fake_scores=True):
    if False:
        print('Hello World!')
    text = '((S (VP (VB Unban)) (NP (NNP Mox) (NNP Opal))))'
    trees = tree_reader.read_trees(text)
    assert len(trees) == 1
    gold = trees[0]
    if fake_scores:
        prediction = (gold, 1.0)
        treebank = [ParseResult(gold, [prediction], None, None)]
        return treebank
    else:
        prediction = gold
        return collate([gold], [prediction])

def check_build(fake_scores=True):
    if False:
        while True:
            i = 10
    treebank = build_one_tree_treebank(fake_scores)
    request = build_request(treebank)
    assert len(request.treebank) == 1
    check_tree(request.treebank[0].gold, treebank[0][0], None)
    assert len(request.treebank[0].predicted) == 1
    if fake_scores:
        check_tree(request.treebank[0].predicted[0], treebank[0][1][0][0], treebank[0][1][0][1])
    else:
        check_tree(request.treebank[0].predicted[0], treebank[0][1][0], None)

def test_build_tuple_request():
    if False:
        i = 10
        return i + 15
    check_build(True)

def test_build_notuple_request():
    if False:
        return 10
    check_build(False)

def test_score_one_tree_tuples():
    if False:
        i = 10
        return i + 15
    treebank = build_one_tree_treebank(True)
    with EvaluateParser() as ep:
        response = ep.process(treebank)
        assert response.f1 == pytest.approx(1.0)

def test_score_one_tree_notuples():
    if False:
        for i in range(10):
            print('nop')
    treebank = build_one_tree_treebank(False)
    with EvaluateParser() as ep:
        response = ep.process(treebank)
        assert response.f1 == pytest.approx(1.0)