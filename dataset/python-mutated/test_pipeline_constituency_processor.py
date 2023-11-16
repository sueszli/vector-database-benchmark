import pytest
import stanza
from stanza.models.common.foundation_cache import FoundationCache
from stanza.tests import *
pytestmark = [pytest.mark.pipeline, pytest.mark.travis]
TEST_TEXT = 'This is a test.  Another sentence.  Are these sorted?'
TEST_TOKENS = [['This', 'is', 'a', 'test', '.'], ['Another', 'sentence', '.'], ['Are', 'these', 'sorted', '?']]

@pytest.fixture(scope='module')
def foundation_cache():
    if False:
        i = 10
        return i + 15
    return FoundationCache()

def check_results(doc):
    if False:
        for i in range(10):
            print('nop')
    assert len(doc.sentences) == len(TEST_TOKENS)
    for (sentence, expected) in zip(doc.sentences, TEST_TOKENS):
        assert sentence.constituency.leaf_labels() == expected

def test_sorted_big_batch(foundation_cache):
    if False:
        for i in range(10):
            print('nop')
    pipe = stanza.Pipeline('en', model_dir=TEST_MODELS_DIR, processors='tokenize,pos,constituency', foundation_cache=foundation_cache)
    doc = pipe(TEST_TEXT)
    check_results(doc)

def test_comments(foundation_cache):
    if False:
        print('Hello World!')
    '\n    Test that the pipeline is creating constituency comments\n    '
    pipe = stanza.Pipeline('en', model_dir=TEST_MODELS_DIR, processors='tokenize,pos,constituency', foundation_cache=foundation_cache)
    doc = pipe(TEST_TEXT)
    check_results(doc)
    for sentence in doc.sentences:
        assert any((x.startswith('# constituency = ') for x in sentence.comments))
    doc.sentences[0].constituency = 'asdf'
    assert '# constituency = asdf' in doc.sentences[0].comments
    for sentence in doc.sentences:
        assert len([x for x in sentence.comments if x.startswith('# constituency')]) == 1

def test_illegal_batch_size(foundation_cache):
    if False:
        i = 10
        return i + 15
    stanza.Pipeline('en', model_dir=TEST_MODELS_DIR, processors='tokenize,pos', constituency_batch_size='zzz', foundation_cache=foundation_cache)
    with pytest.raises(ValueError):
        stanza.Pipeline('en', model_dir=TEST_MODELS_DIR, processors='tokenize,pos,constituency', constituency_batch_size='zzz', foundation_cache=foundation_cache)

def test_sorted_one_batch(foundation_cache):
    if False:
        return 10
    pipe = stanza.Pipeline('en', model_dir=TEST_MODELS_DIR, processors='tokenize,pos,constituency', constituency_batch_size=1, foundation_cache=foundation_cache)
    doc = pipe(TEST_TEXT)
    check_results(doc)

def test_sorted_two_batch(foundation_cache):
    if False:
        return 10
    pipe = stanza.Pipeline('en', model_dir=TEST_MODELS_DIR, processors='tokenize,pos,constituency', constituency_batch_size=2, foundation_cache=foundation_cache)
    doc = pipe(TEST_TEXT)
    check_results(doc)

def test_get_constituents(foundation_cache):
    if False:
        for i in range(10):
            print('nop')
    pipe = stanza.Pipeline('en', processors='tokenize,pos,constituency', foundation_cache=foundation_cache)
    assert 'SBAR' in pipe.processors['constituency'].get_constituents()