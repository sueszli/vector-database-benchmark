"""
Tests to read a stored protobuf.
Also serves as an example of how to parse sentences, tokens, pos, lemma,
ner, dependencies and mentions.

The test corresponds to annotations for the following sentence:
    Chris wrote a simple sentence that he parsed with Stanford CoreNLP.
"""
import os
from pathlib import Path
import pytest
from pytest import fixture
from stanza.protobuf import Document, Sentence, Token, DependencyGraph, CorefChain
from stanza.protobuf import parseFromDelimitedString, writeToDelimitedString, to_text
pytestmark = [pytest.mark.travis, pytest.mark.client]
TEXT = 'Chris wrote a simple sentence that he parsed with Stanford CoreNLP.\n'

@fixture
def doc_pb():
    if False:
        print('Hello World!')
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = Path(test_dir).parent
    test_data = os.path.join(test_dir, 'data', 'test.dat')
    with open(test_data, 'rb') as f:
        buf = f.read()
    doc = Document()
    parseFromDelimitedString(doc, buf)
    return doc

def test_parse_protobuf(doc_pb):
    if False:
        i = 10
        return i + 15
    assert doc_pb.ByteSize() == 4709

def test_write_protobuf(doc_pb):
    if False:
        return 10
    stream = writeToDelimitedString(doc_pb)
    buf = stream.getvalue()
    stream.close()
    doc_pb_ = Document()
    parseFromDelimitedString(doc_pb_, buf)
    assert doc_pb == doc_pb_

def test_document_text(doc_pb):
    if False:
        return 10
    assert doc_pb.text == TEXT

def test_sentences(doc_pb):
    if False:
        while True:
            i = 10
    assert len(doc_pb.sentence) == 1
    sentence = doc_pb.sentence[0]
    assert isinstance(sentence, Sentence)
    assert sentence.characterOffsetEnd - sentence.characterOffsetBegin == 67
    assert sentence.text == ''
    assert to_text(sentence) == TEXT[:-1]

def test_tokens(doc_pb):
    if False:
        while True:
            i = 10
    sentence = doc_pb.sentence[0]
    tokens = sentence.token
    assert len(tokens) == 12
    assert isinstance(tokens[0], Token)
    words = 'Chris wrote a simple sentence that he parsed with Stanford CoreNLP .'.split()
    words_ = [t.word for t in tokens]
    assert words_ == words
    lemmas = 'Chris write a simple sentence that he parse with Stanford CoreNLP .'.split()
    lemmas_ = [t.lemma for t in tokens]
    assert lemmas_ == lemmas
    pos = 'NNP VBD DT JJ NN IN PRP VBD IN NNP NNP .'.split()
    pos_ = [t.pos for t in tokens]
    assert pos_ == pos
    ner = 'PERSON O O O O O O O O ORGANIZATION O O'.split()
    ner_ = [t.ner for t in tokens]
    assert ner_ == ner
    begin = [int(i) for i in '0 6 12 14 21 30 35 38 45 50 59 66'.split()]
    end = [int(i) for i in '5 11 13 20 29 34 37 44 49 58 66 67'.split()]
    begin_ = [t.beginChar for t in tokens]
    end_ = [t.endChar for t in tokens]
    assert begin_ == begin
    assert end_ == end

def test_dependency_parse(doc_pb):
    if False:
        for i in range(10):
            print('nop')
    '\n    Extract the dependency parse from the annotation.\n    '
    sentence = doc_pb.sentence[0]
    assert sentence.basicDependencies.ByteSize() > 0
    assert sentence.enhancedDependencies.ByteSize() > 0
    assert sentence.enhancedPlusPlusDependencies.ByteSize() > 0
    tree = sentence.enhancedPlusPlusDependencies
    isinstance(tree, DependencyGraph)
    assert tree.root
    assert len(tree.node) == len(sentence.token)
    assert len(tree.edge) == 12
    edge = tree.edge[0]
    assert edge.source == 2
    assert edge.target == 1
    assert edge.dep == 'nsubj'

def test_coref_chain(doc_pb):
    if False:
        while True:
            i = 10
    '\n    Extract the corefence chains from the annotation.\n    '
    chains = doc_pb.corefChain
    assert len(chains) == 1
    chain = chains[0]
    assert isinstance(chain, CorefChain)
    assert chain.mention[0].beginIndex == 0
    assert chain.mention[0].endIndex == 1
    assert chain.mention[0].gender == 'MALE'
    assert chain.mention[1].beginIndex == 6
    assert chain.mention[1].endIndex == 7
    assert chain.mention[1].gender == 'MALE'
    assert chain.representative == 0