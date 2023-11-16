import hypothesis
import hypothesis.strategies
import numpy
import pytest
from thinc.tests.strategies import ndarrays_of_shape
from spacy.language import Language
from spacy.pipeline._parser_internals._beam_utils import BeamBatch
from spacy.pipeline._parser_internals.arc_eager import ArcEager
from spacy.pipeline._parser_internals.stateclass import StateClass
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab

@pytest.fixture(scope='module')
def vocab():
    if False:
        while True:
            i = 10
    return Vocab()

@pytest.fixture(scope='module')
def moves(vocab):
    if False:
        for i in range(10):
            print('nop')
    aeager = ArcEager(vocab.strings, {})
    aeager.add_action(0, '')
    aeager.add_action(1, '')
    aeager.add_action(2, 'nsubj')
    aeager.add_action(2, 'punct')
    aeager.add_action(2, 'aux')
    aeager.add_action(2, 'nsubjpass')
    aeager.add_action(3, 'dobj')
    aeager.add_action(2, 'aux')
    aeager.add_action(4, 'ROOT')
    return aeager

@pytest.fixture(scope='module')
def docs(vocab):
    if False:
        i = 10
        return i + 15
    return [Doc(vocab, words=['Rats', 'bite', 'things'], heads=[1, 1, 1], deps=['nsubj', 'ROOT', 'dobj'], sent_starts=[True, False, False])]

@pytest.fixture(scope='module')
def examples(docs):
    if False:
        print('Hello World!')
    return [Example(doc, doc.copy()) for doc in docs]

@pytest.fixture
def states(docs):
    if False:
        i = 10
        return i + 15
    return [StateClass(doc) for doc in docs]

@pytest.fixture
def tokvecs(docs, vector_size):
    if False:
        for i in range(10):
            print('nop')
    output = []
    for doc in docs:
        vec = numpy.random.uniform(-0.1, 0.1, (len(doc), vector_size))
        output.append(numpy.asarray(vec))
    return output

@pytest.fixture(scope='module')
def batch_size(docs):
    if False:
        for i in range(10):
            print('nop')
    return len(docs)

@pytest.fixture(scope='module')
def beam_width():
    if False:
        i = 10
        return i + 15
    return 4

@pytest.fixture(params=[0.0, 0.5, 1.0])
def beam_density(request):
    if False:
        return 10
    return request.param

@pytest.fixture
def vector_size():
    if False:
        return 10
    return 6

@pytest.fixture
def beam(moves, examples, beam_width):
    if False:
        while True:
            i = 10
    (states, golds, _) = moves.init_gold_batch(examples)
    return BeamBatch(moves, states, golds, width=beam_width, density=0.0)

@pytest.fixture
def scores(moves, batch_size, beam_width):
    if False:
        i = 10
        return i + 15
    return numpy.asarray(numpy.concatenate([numpy.random.uniform(-0.1, 0.1, (beam_width, moves.n_moves)) for _ in range(batch_size)]), dtype='float32')

def test_create_beam(beam):
    if False:
        i = 10
        return i + 15
    pass

def test_beam_advance(beam, scores):
    if False:
        i = 10
        return i + 15
    beam.advance(scores)

def test_beam_advance_too_few_scores(beam, scores):
    if False:
        while True:
            i = 10
    n_state = sum((len(beam) for beam in beam))
    scores = scores[:n_state]
    with pytest.raises(IndexError):
        beam.advance(scores[:-1])

def test_beam_parse(examples, beam_width):
    if False:
        while True:
            i = 10
    nlp = Language()
    parser = nlp.add_pipe('beam_parser')
    parser.cfg['beam_width'] = beam_width
    parser.add_label('nsubj')
    parser.initialize(lambda : examples)
    doc = nlp.make_doc('Australia is a country')
    parser(doc)

@hypothesis.given(hyp=hypothesis.strategies.data())
def test_beam_density(moves, examples, beam_width, hyp):
    if False:
        i = 10
        return i + 15
    beam_density = float(hyp.draw(hypothesis.strategies.floats(0.0, 1.0, width=32)))
    (states, golds, _) = moves.init_gold_batch(examples)
    beam = BeamBatch(moves, states, golds, width=beam_width, density=beam_density)
    n_state = sum((len(beam) for beam in beam))
    scores = hyp.draw(ndarrays_of_shape((n_state, moves.n_moves)))
    beam.advance(scores)
    for b in beam:
        beam_probs = b.probs
        assert b.min_density == beam_density
        assert beam_probs[-1] >= beam_probs[0] * beam_density