from typing import Any, Callable, Dict, Iterable, Tuple
import pytest
from numpy.testing import assert_equal
from spacy import Language, registry, util
from spacy.attrs import ENT_KB_ID
from spacy.compat import pickle
from spacy.kb import Candidate, InMemoryLookupKB, KnowledgeBase, get_candidates
from spacy.lang.en import English
from spacy.ml import load_kb
from spacy.ml.models.entity_linker import build_span_maker
from spacy.pipeline import EntityLinker
from spacy.pipeline.legacy import EntityLinker_v1
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.scorer import Scorer
from spacy.tests.util import make_tempdir
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import ensure_path
from spacy.vocab import Vocab

@pytest.fixture
def nlp():
    if False:
        for i in range(10):
            print('nop')
    return English()

def assert_almost_equal(a, b):
    if False:
        i = 10
        return i + 15
    delta = 0.0001
    assert a - delta <= b <= a + delta

@pytest.mark.issue(4674)
def test_issue4674():
    if False:
        i = 10
        return i + 15
    'Test that setting entities with overlapping identifiers does not mess up IO'
    nlp = English()
    kb = InMemoryLookupKB(nlp.vocab, entity_vector_length=3)
    vector1 = [0.9, 1.1, 1.01]
    vector2 = [1.8, 2.25, 2.01]
    with pytest.warns(UserWarning):
        kb.set_entities(entity_list=['Q1', 'Q1'], freq_list=[32, 111], vector_list=[vector1, vector2])
    assert kb.get_size_entities() == 1
    with make_tempdir() as d:
        dir_path = ensure_path(d)
        if not dir_path.exists():
            dir_path.mkdir()
        file_path = dir_path / 'kb'
        kb.to_disk(str(file_path))
        kb2 = InMemoryLookupKB(nlp.vocab, entity_vector_length=3)
        kb2.from_disk(str(file_path))
    assert kb2.get_size_entities() == 1

@pytest.mark.issue(6730)
def test_issue6730(en_vocab):
    if False:
        return 10
    'Ensure that the KB does not accept empty strings, but otherwise IO works fine.'
    from spacy.kb.kb_in_memory import InMemoryLookupKB
    kb = InMemoryLookupKB(en_vocab, entity_vector_length=3)
    kb.add_entity(entity='1', freq=148, entity_vector=[1, 2, 3])
    with pytest.raises(ValueError):
        kb.add_alias(alias='', entities=['1'], probabilities=[0.4])
    assert kb.contains_alias('') is False
    kb.add_alias(alias='x', entities=['1'], probabilities=[0.2])
    kb.add_alias(alias='y', entities=['1'], probabilities=[0.1])
    with make_tempdir() as tmp_dir:
        kb.to_disk(tmp_dir)
        kb.from_disk(tmp_dir)
    assert kb.get_size_aliases() == 2
    assert set(kb.get_alias_strings()) == {'x', 'y'}

@pytest.mark.issue(7065)
def test_issue7065():
    if False:
        while True:
            i = 10
    text = "Kathleen Battle sang in Mahler 's Symphony No. 8 at the Cincinnati Symphony Orchestra 's May Festival."
    nlp = English()
    nlp.add_pipe('sentencizer')
    ruler = nlp.add_pipe('entity_ruler')
    patterns = [{'label': 'THING', 'pattern': [{'LOWER': 'symphony'}, {'LOWER': 'no'}, {'LOWER': '.'}, {'LOWER': '8'}]}]
    ruler.add_patterns(patterns)
    doc = nlp(text)
    sentences = [s for s in doc.sents]
    assert len(sentences) == 2
    sent0 = sentences[0]
    ent = doc.ents[0]
    assert ent.start < sent0.end < ent.end
    assert sentences.index(ent.sent) == 0

@pytest.mark.issue(7065)
@pytest.mark.parametrize('entity_in_first_sentence', [True, False])
def test_sentence_crossing_ents(entity_in_first_sentence: bool):
    if False:
        i = 10
        return i + 15
    "Tests if NEL crashes if entities cross sentence boundaries and the first associated sentence doesn't have an\n    entity.\n    entity_in_prior_sentence (bool): Whether to include an entity in the first sentence associated with the\n    sentence-crossing entity.\n    "
    nlp = English()
    vector_length = 3
    text = "Mahler 's Symphony No. 8 was beautiful."
    entities = [(10, 24, 'WORK')]
    links = {(10, 24): {'Q7304': 0.0, 'Q270853': 1.0}}
    if entity_in_first_sentence:
        entities.append((0, 6, 'PERSON'))
        links[0, 6] = {'Q7304': 1.0, 'Q270853': 0.0}
    sent_starts = [1, -1, 0, 0, 0, 1, 0, 0, 0]
    doc = nlp(text)
    example = Example.from_dict(doc, {'entities': entities, 'links': links, 'sent_starts': sent_starts})
    train_examples = [example]

    def create_kb(vocab):
        if False:
            for i in range(10):
                print('nop')
        mykb = InMemoryLookupKB(vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q270853', freq=12, entity_vector=[9, 1, -7])
        mykb.add_alias(alias='No. 8', entities=['Q270853'], probabilities=[1.0])
        mykb.add_entity(entity='Q7304', freq=12, entity_vector=[6, -4, 3])
        mykb.add_alias(alias='Mahler', entities=['Q7304'], probabilities=[1.0])
        return mykb
    entity_linker = nlp.add_pipe('entity_linker', last=True)
    entity_linker.set_kb(create_kb)
    optimizer = nlp.initialize(get_examples=lambda : train_examples)
    for i in range(2):
        nlp.update(train_examples, sgd=optimizer)
    entity_linker.predict([example.reference])

def test_no_entities():
    if False:
        return 10
    TRAIN_DATA = [('The sky is blue.', {'sent_starts': [1, 0, 0, 0, 0]})]
    nlp = English()
    vector_length = 3
    train_examples = []
    for (text, annotation) in TRAIN_DATA:
        doc = nlp(text)
        train_examples.append(Example.from_dict(doc, annotation))

    def create_kb(vocab):
        if False:
            while True:
                i = 10
        mykb = InMemoryLookupKB(vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q2146908', freq=12, entity_vector=[6, -4, 3])
        mykb.add_alias('Russ Cochran', ['Q2146908'], [0.9])
        return mykb
    entity_linker = nlp.add_pipe('entity_linker', last=True)
    entity_linker.set_kb(create_kb)
    optimizer = nlp.initialize(get_examples=lambda : train_examples)
    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    nlp.add_pipe('sentencizer', first=True)
    nlp.evaluate(train_examples)

def test_partial_links():
    if False:
        print('Hello World!')
    TRAIN_DATA = [('Russ Cochran his reprints include EC Comics.', {'links': {(0, 12): {'Q2146908': 1.0}}, 'entities': [(0, 12, 'PERSON')], 'sent_starts': [1, -1, 0, 0, 0, 0, 0, 0]})]
    nlp = English()
    vector_length = 3
    train_examples = []
    for (text, annotation) in TRAIN_DATA:
        doc = nlp(text)
        train_examples.append(Example.from_dict(doc, annotation))

    def create_kb(vocab):
        if False:
            return 10
        mykb = InMemoryLookupKB(vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q2146908', freq=12, entity_vector=[6, -4, 3])
        mykb.add_alias('Russ Cochran', ['Q2146908'], [0.9])
        return mykb
    entity_linker = nlp.add_pipe('entity_linker', last=True)
    entity_linker.set_kb(create_kb)
    optimizer = nlp.initialize(get_examples=lambda : train_examples)
    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    nlp.add_pipe('sentencizer', first=True)
    patterns = [{'label': 'PERSON', 'pattern': [{'LOWER': 'russ'}, {'LOWER': 'cochran'}]}, {'label': 'ORG', 'pattern': [{'LOWER': 'ec'}, {'LOWER': 'comics'}]}]
    ruler = nlp.add_pipe('entity_ruler', before='entity_linker')
    ruler.add_patterns(patterns)
    results = nlp.evaluate(train_examples)
    assert 'PERSON' in results['ents_per_type']
    assert 'PERSON' in results['nel_f_per_type']
    assert 'ORG' in results['ents_per_type']
    assert 'ORG' not in results['nel_f_per_type']

def test_kb_valid_entities(nlp):
    if False:
        while True:
            i = 10
    'Test the valid construction of a KB with 3 entities and two aliases'
    mykb = InMemoryLookupKB(nlp.vocab, entity_vector_length=3)
    mykb.add_entity(entity='Q1', freq=19, entity_vector=[8, 4, 3])
    mykb.add_entity(entity='Q2', freq=5, entity_vector=[2, 1, 0])
    mykb.add_entity(entity='Q3', freq=25, entity_vector=[-1, -6, 5])
    mykb.add_alias(alias='douglas', entities=['Q2', 'Q3'], probabilities=[0.8, 0.2])
    mykb.add_alias(alias='adam', entities=['Q2'], probabilities=[0.9])
    assert mykb.get_size_entities() == 3
    assert mykb.get_size_aliases() == 2
    assert mykb.get_vector('Q1') == [8, 4, 3]
    assert mykb.get_vector('Q2') == [2, 1, 0]
    assert mykb.get_vector('Q3') == [-1, -6, 5]
    assert_almost_equal(mykb.get_prior_prob(entity='Q2', alias='douglas'), 0.8)
    assert_almost_equal(mykb.get_prior_prob(entity='Q3', alias='douglas'), 0.2)
    assert_almost_equal(mykb.get_prior_prob(entity='Q342', alias='douglas'), 0.0)
    assert_almost_equal(mykb.get_prior_prob(entity='Q3', alias='douglassssss'), 0.0)

def test_kb_invalid_entities(nlp):
    if False:
        i = 10
        return i + 15
    'Test the invalid construction of a KB with an alias linked to a non-existing entity'
    mykb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
    mykb.add_entity(entity='Q1', freq=19, entity_vector=[1])
    mykb.add_entity(entity='Q2', freq=5, entity_vector=[2])
    mykb.add_entity(entity='Q3', freq=25, entity_vector=[3])
    with pytest.raises(ValueError):
        mykb.add_alias(alias='douglas', entities=['Q2', 'Q342'], probabilities=[0.8, 0.2])

def test_kb_invalid_probabilities(nlp):
    if False:
        return 10
    'Test the invalid construction of a KB with wrong prior probabilities'
    mykb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
    mykb.add_entity(entity='Q1', freq=19, entity_vector=[1])
    mykb.add_entity(entity='Q2', freq=5, entity_vector=[2])
    mykb.add_entity(entity='Q3', freq=25, entity_vector=[3])
    with pytest.raises(ValueError):
        mykb.add_alias(alias='douglas', entities=['Q2', 'Q3'], probabilities=[0.8, 0.4])

def test_kb_invalid_combination(nlp):
    if False:
        for i in range(10):
            print('nop')
    'Test the invalid construction of a KB with non-matching entity and probability lists'
    mykb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
    mykb.add_entity(entity='Q1', freq=19, entity_vector=[1])
    mykb.add_entity(entity='Q2', freq=5, entity_vector=[2])
    mykb.add_entity(entity='Q3', freq=25, entity_vector=[3])
    with pytest.raises(ValueError):
        mykb.add_alias(alias='douglas', entities=['Q2', 'Q3'], probabilities=[0.3, 0.4, 0.1])

def test_kb_invalid_entity_vector(nlp):
    if False:
        return 10
    'Test the invalid construction of a KB with non-matching entity vector lengths'
    mykb = InMemoryLookupKB(nlp.vocab, entity_vector_length=3)
    mykb.add_entity(entity='Q1', freq=19, entity_vector=[1, 2, 3])
    with pytest.raises(ValueError):
        mykb.add_entity(entity='Q2', freq=5, entity_vector=[2])

def test_kb_default(nlp):
    if False:
        i = 10
        return i + 15
    'Test that the default (empty) KB is loaded upon construction'
    entity_linker = nlp.add_pipe('entity_linker', config={})
    assert len(entity_linker.kb) == 0
    with pytest.raises(ValueError, match='E139'):
        entity_linker.validate_kb()
    assert entity_linker.kb.get_size_entities() == 0
    assert entity_linker.kb.get_size_aliases() == 0
    assert entity_linker.kb.entity_vector_length == 64

def test_kb_custom_length(nlp):
    if False:
        for i in range(10):
            print('nop')
    'Test that the default (empty) KB can be configured with a custom entity length'
    entity_linker = nlp.add_pipe('entity_linker', config={'entity_vector_length': 35})
    assert len(entity_linker.kb) == 0
    assert entity_linker.kb.get_size_entities() == 0
    assert entity_linker.kb.get_size_aliases() == 0
    assert entity_linker.kb.entity_vector_length == 35

def test_kb_initialize_empty(nlp):
    if False:
        return 10
    "Test that the EL can't initialize without examples"
    entity_linker = nlp.add_pipe('entity_linker')
    with pytest.raises(TypeError):
        entity_linker.initialize(lambda : [])

def test_kb_serialize(nlp):
    if False:
        i = 10
        return i + 15
    'Test serialization of the KB'
    mykb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
    with make_tempdir() as d:
        mykb.to_disk(d / 'kb')
        mykb.from_disk(d / 'kb')
        mykb.to_disk(d / 'new' / 'kb')
        mykb.from_disk(d / 'new' / 'kb')
        mykb.to_disk(d / 'kb')
        with pytest.raises(ValueError):
            mykb.from_disk(d / 'unknown' / 'kb')

@pytest.mark.issue(9137)
def test_kb_serialize_2(nlp):
    if False:
        i = 10
        return i + 15
    v = [5, 6, 7, 8]
    kb1 = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=4)
    kb1.set_entities(['E1'], [1], [v])
    assert kb1.get_vector('E1') == v
    with make_tempdir() as d:
        kb1.to_disk(d / 'kb')
        kb2 = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=4)
        kb2.from_disk(d / 'kb')
        assert kb2.get_vector('E1') == v

def test_kb_set_entities(nlp):
    if False:
        i = 10
        return i + 15
    'Test that set_entities entirely overwrites the previous set of entities'
    v = [5, 6, 7, 8]
    v1 = [1, 1, 1, 0]
    v2 = [2, 2, 2, 3]
    kb1 = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=4)
    kb1.set_entities(['E0'], [1], [v])
    assert kb1.get_entity_strings() == ['E0']
    kb1.set_entities(['E1', 'E2'], [1, 9], [v1, v2])
    assert set(kb1.get_entity_strings()) == {'E1', 'E2'}
    assert kb1.get_vector('E1') == v1
    assert kb1.get_vector('E2') == v2
    with make_tempdir() as d:
        kb1.to_disk(d / 'kb')
        kb2 = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=4)
        kb2.from_disk(d / 'kb')
        assert set(kb2.get_entity_strings()) == {'E1', 'E2'}
        assert kb2.get_vector('E1') == v1
        assert kb2.get_vector('E2') == v2

def test_kb_serialize_vocab(nlp):
    if False:
        print('Hello World!')
    'Test serialization of the KB and custom strings'
    entity = 'MyFunnyID'
    assert entity not in nlp.vocab.strings
    mykb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
    assert not mykb.contains_entity(entity)
    mykb.add_entity(entity, freq=342, entity_vector=[3])
    assert mykb.contains_entity(entity)
    assert entity in mykb.vocab.strings
    with make_tempdir() as d:
        mykb.to_disk(d / 'kb')
        mykb_new = InMemoryLookupKB(Vocab(), entity_vector_length=1)
        mykb_new.from_disk(d / 'kb')
        assert entity in mykb_new.vocab.strings

def test_candidate_generation(nlp):
    if False:
        print('Hello World!')
    'Test correct candidate generation'
    mykb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
    doc = nlp('douglas adam Adam shrubbery')
    douglas_ent = doc[0:1]
    adam_ent = doc[1:2]
    Adam_ent = doc[2:3]
    shrubbery_ent = doc[3:4]
    mykb.add_entity(entity='Q1', freq=27, entity_vector=[1])
    mykb.add_entity(entity='Q2', freq=12, entity_vector=[2])
    mykb.add_entity(entity='Q3', freq=5, entity_vector=[3])
    mykb.add_alias(alias='douglas', entities=['Q2', 'Q3'], probabilities=[0.8, 0.1])
    mykb.add_alias(alias='adam', entities=['Q2'], probabilities=[0.9])
    assert len(get_candidates(mykb, douglas_ent)) == 2
    assert len(get_candidates(mykb, adam_ent)) == 1
    assert len(get_candidates(mykb, Adam_ent)) == 0
    assert len(get_candidates(mykb, shrubbery_ent)) == 0
    assert get_candidates(mykb, adam_ent)[0].entity_ == 'Q2'
    assert get_candidates(mykb, adam_ent)[0].alias_ == 'adam'
    assert_almost_equal(get_candidates(mykb, adam_ent)[0].entity_freq, 12)
    assert_almost_equal(get_candidates(mykb, adam_ent)[0].prior_prob, 0.9)

def test_el_pipe_configuration(nlp):
    if False:
        return 10
    'Test correct candidate generation as part of the EL pipe'
    nlp.add_pipe('sentencizer')
    pattern = {'label': 'PERSON', 'pattern': [{'LOWER': 'douglas'}]}
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns([pattern])

    def create_kb(vocab):
        if False:
            return 10
        kb = InMemoryLookupKB(vocab, entity_vector_length=1)
        kb.add_entity(entity='Q2', freq=12, entity_vector=[2])
        kb.add_entity(entity='Q3', freq=5, entity_vector=[3])
        kb.add_alias(alias='douglas', entities=['Q2', 'Q3'], probabilities=[0.8, 0.1])
        return kb
    entity_linker = nlp.add_pipe('entity_linker', config={'incl_context': False})
    entity_linker.set_kb(create_kb)
    text = 'Douglas and douglas are not the same.'
    doc = nlp(text)
    assert doc[0].ent_kb_id_ == 'NIL'
    assert doc[1].ent_kb_id_ == ''
    assert doc[2].ent_kb_id_ == 'Q2'

    def get_lowercased_candidates(kb, span):
        if False:
            while True:
                i = 10
        return kb.get_alias_candidates(span.text.lower())

    def get_lowercased_candidates_batch(kb, spans):
        if False:
            for i in range(10):
                print('nop')
        return [get_lowercased_candidates(kb, span) for span in spans]

    @registry.misc('spacy.LowercaseCandidateGenerator.v1')
    def create_candidates() -> Callable[[InMemoryLookupKB, 'Span'], Iterable[Candidate]]:
        if False:
            i = 10
            return i + 15
        return get_lowercased_candidates

    @registry.misc('spacy.LowercaseCandidateBatchGenerator.v1')
    def create_candidates_batch() -> Callable[[InMemoryLookupKB, Iterable['Span']], Iterable[Iterable[Candidate]]]:
        if False:
            return 10
        return get_lowercased_candidates_batch
    entity_linker = nlp.replace_pipe('entity_linker', 'entity_linker', config={'incl_context': False, 'get_candidates': {'@misc': 'spacy.LowercaseCandidateGenerator.v1'}, 'get_candidates_batch': {'@misc': 'spacy.LowercaseCandidateBatchGenerator.v1'}})
    entity_linker.set_kb(create_kb)
    doc = nlp(text)
    assert doc[0].ent_kb_id_ == 'Q2'
    assert doc[1].ent_kb_id_ == ''
    assert doc[2].ent_kb_id_ == 'Q2'

def test_nel_nsents(nlp):
    if False:
        return 10
    'Test that n_sents can be set through the configuration'
    entity_linker = nlp.add_pipe('entity_linker', config={})
    assert entity_linker.n_sents == 0
    entity_linker = nlp.replace_pipe('entity_linker', 'entity_linker', config={'n_sents': 2})
    assert entity_linker.n_sents == 2

def test_vocab_serialization(nlp):
    if False:
        return 10
    'Test that string information is retained across storage'
    mykb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
    mykb.add_entity(entity='Q1', freq=27, entity_vector=[1])
    q2_hash = mykb.add_entity(entity='Q2', freq=12, entity_vector=[2])
    mykb.add_entity(entity='Q3', freq=5, entity_vector=[3])
    mykb.add_alias(alias='douglas', entities=['Q2', 'Q3'], probabilities=[0.4, 0.1])
    adam_hash = mykb.add_alias(alias='adam', entities=['Q2'], probabilities=[0.9])
    candidates = mykb.get_alias_candidates('adam')
    assert len(candidates) == 1
    assert candidates[0].entity == q2_hash
    assert candidates[0].entity_ == 'Q2'
    assert candidates[0].alias == adam_hash
    assert candidates[0].alias_ == 'adam'
    with make_tempdir() as d:
        mykb.to_disk(d / 'kb')
        kb_new_vocab = InMemoryLookupKB(Vocab(), entity_vector_length=1)
        kb_new_vocab.from_disk(d / 'kb')
        candidates = kb_new_vocab.get_alias_candidates('adam')
        assert len(candidates) == 1
        assert candidates[0].entity == q2_hash
        assert candidates[0].entity_ == 'Q2'
        assert candidates[0].alias == adam_hash
        assert candidates[0].alias_ == 'adam'
        assert kb_new_vocab.get_vector('Q2') == [2]
        assert_almost_equal(kb_new_vocab.get_prior_prob('Q2', 'douglas'), 0.4)

def test_append_alias(nlp):
    if False:
        while True:
            i = 10
    'Test that we can append additional alias-entity pairs'
    mykb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
    mykb.add_entity(entity='Q1', freq=27, entity_vector=[1])
    mykb.add_entity(entity='Q2', freq=12, entity_vector=[2])
    mykb.add_entity(entity='Q3', freq=5, entity_vector=[3])
    mykb.add_alias(alias='douglas', entities=['Q2', 'Q3'], probabilities=[0.4, 0.1])
    mykb.add_alias(alias='adam', entities=['Q2'], probabilities=[0.9])
    assert len(mykb.get_alias_candidates('douglas')) == 2
    mykb.append_alias(alias='douglas', entity='Q1', prior_prob=0.2)
    assert len(mykb.get_alias_candidates('douglas')) == 3
    with pytest.warns(UserWarning):
        mykb.append_alias(alias='douglas', entity='Q1', prior_prob=0.3)
    assert len(mykb.get_alias_candidates('douglas')) == 3

@pytest.mark.filterwarnings('ignore:\\[W036')
def test_append_invalid_alias(nlp):
    if False:
        print('Hello World!')
    'Test that append an alias will throw an error if prior probs are exceeding 1'
    mykb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
    mykb.add_entity(entity='Q1', freq=27, entity_vector=[1])
    mykb.add_entity(entity='Q2', freq=12, entity_vector=[2])
    mykb.add_entity(entity='Q3', freq=5, entity_vector=[3])
    mykb.add_alias(alias='douglas', entities=['Q2', 'Q3'], probabilities=[0.8, 0.1])
    mykb.add_alias(alias='adam', entities=['Q2'], probabilities=[0.9])
    with pytest.raises(ValueError):
        mykb.append_alias(alias='douglas', entity='Q1', prior_prob=0.2)

@pytest.mark.filterwarnings('ignore:\\[W036')
def test_preserving_links_asdoc(nlp):
    if False:
        return 10
    'Test that Span.as_doc preserves the existing entity links'
    vector_length = 1

    def create_kb(vocab):
        if False:
            for i in range(10):
                print('nop')
        mykb = InMemoryLookupKB(vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q1', freq=19, entity_vector=[1])
        mykb.add_entity(entity='Q2', freq=8, entity_vector=[1])
        mykb.add_alias(alias='Boston', entities=['Q1'], probabilities=[0.7])
        mykb.add_alias(alias='Denver', entities=['Q2'], probabilities=[0.6])
        return mykb
    nlp.add_pipe('sentencizer')
    patterns = [{'label': 'GPE', 'pattern': 'Boston'}, {'label': 'GPE', 'pattern': 'Denver'}]
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns(patterns)
    config = {'incl_prior': False}
    entity_linker = nlp.add_pipe('entity_linker', config=config, last=True)
    entity_linker.set_kb(create_kb)
    nlp.initialize()
    assert entity_linker.model.get_dim('nO') == vector_length
    text = 'She lives in Boston. He lives in Denver.'
    doc = nlp(text)
    for ent in doc.ents:
        orig_text = ent.text
        orig_kb_id = ent.kb_id_
        sent_doc = ent.sent.as_doc()
        for s_ent in sent_doc.ents:
            if s_ent.text == orig_text:
                assert s_ent.kb_id_ == orig_kb_id

def test_preserving_links_ents(nlp):
    if False:
        print('Hello World!')
    'Test that doc.ents preserves KB annotations'
    text = 'She lives in Boston. He lives in Denver.'
    doc = nlp(text)
    assert len(list(doc.ents)) == 0
    boston_ent = Span(doc, 3, 4, label='LOC', kb_id='Q1')
    doc.ents = [boston_ent]
    assert len(list(doc.ents)) == 1
    assert list(doc.ents)[0].label_ == 'LOC'
    assert list(doc.ents)[0].kb_id_ == 'Q1'

def test_preserving_links_ents_2(nlp):
    if False:
        while True:
            i = 10
    'Test that doc.ents preserves KB annotations'
    text = 'She lives in Boston. He lives in Denver.'
    doc = nlp(text)
    assert len(list(doc.ents)) == 0
    loc = doc.vocab.strings.add('LOC')
    q1 = doc.vocab.strings.add('Q1')
    doc.ents = [(loc, q1, 3, 4)]
    assert len(list(doc.ents)) == 1
    assert list(doc.ents)[0].label_ == 'LOC'
    assert list(doc.ents)[0].kb_id_ == 'Q1'
TRAIN_DATA = [('Russ Cochran captured his first major title with his son as caddie.', {'links': {(0, 12): {'Q7381115': 0.0, 'Q2146908': 1.0}}, 'entities': [(0, 12, 'PERSON')], 'sent_starts': [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}), ('Russ Cochran his reprints include EC Comics.', {'links': {(0, 12): {'Q7381115': 1.0, 'Q2146908': 0.0}}, 'entities': [(0, 12, 'PERSON'), (34, 43, 'ART')], 'sent_starts': [1, -1, 0, 0, 0, 0, 0, 0]}), ('Russ Cochran has been publishing comic art.', {'links': {(0, 12): {'Q7381115': 1.0, 'Q2146908': 0.0}}, 'entities': [(0, 12, 'PERSON')], 'sent_starts': [1, -1, 0, 0, 0, 0, 0, 0]}), ("Russ Cochran was a member of University of Kentucky's golf team.", {'links': {(0, 12): {'Q7381115': 0.0, 'Q2146908': 1.0}}, 'entities': [(0, 12, 'PERSON'), (43, 51, 'LOC')], 'sent_starts': [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}), ('The weather is nice today.', {'links': {}, 'entities': [], 'sent_starts': [1, -1, 0, 0, 0, 0]})]
GOLD_entities = ['Q2146908', 'Q7381115', 'Q7381115', 'Q2146908']

def test_overfitting_IO():
    if False:
        while True:
            i = 10
    nlp = English()
    vector_length = 3
    assert 'Q2146908' not in nlp.vocab.strings
    train_examples = []
    for (text, annotation) in TRAIN_DATA:
        doc = nlp(text)
        train_examples.append(Example.from_dict(doc, annotation))

    def create_kb(vocab):
        if False:
            print('Hello World!')
        mykb = InMemoryLookupKB(vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q2146908', freq=12, entity_vector=[6, -4, 3])
        mykb.add_entity(entity='Q7381115', freq=12, entity_vector=[9, 1, -7])
        mykb.add_alias(alias='Russ Cochran', entities=['Q2146908', 'Q7381115'], probabilities=[0.5, 0.5])
        return mykb
    entity_linker = nlp.add_pipe('entity_linker', last=True)
    assert isinstance(entity_linker, EntityLinker)
    entity_linker.set_kb(create_kb)
    assert 'Q2146908' in entity_linker.vocab.strings
    assert 'Q2146908' in entity_linker.kb.vocab.strings
    optimizer = nlp.initialize(get_examples=lambda : train_examples)
    assert entity_linker.model.get_dim('nO') == vector_length
    assert entity_linker.model.get_dim('nO') == entity_linker.kb.entity_vector_length
    for i in range(50):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses['entity_linker'] < 0.001
    nlp.add_pipe('sentencizer', first=True)
    patterns = [{'label': 'PERSON', 'pattern': [{'LOWER': 'russ'}, {'LOWER': 'cochran'}]}]
    ruler = nlp.add_pipe('entity_ruler', before='entity_linker')
    ruler.add_patterns(patterns)
    predictions = []
    for (text, annotation) in TRAIN_DATA:
        doc = nlp(text)
        for ent in doc.ents:
            predictions.append(ent.kb_id_)
    assert predictions == GOLD_entities
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        assert nlp2.pipe_names == nlp.pipe_names
        assert 'Q2146908' in nlp2.vocab.strings
        entity_linker2 = nlp2.get_pipe('entity_linker')
        assert 'Q2146908' in entity_linker2.vocab.strings
        assert 'Q2146908' in entity_linker2.kb.vocab.strings
        predictions = []
        for (text, annotation) in TRAIN_DATA:
            doc2 = nlp2(text)
            for ent in doc2.ents:
                predictions.append(ent.kb_id_)
        assert predictions == GOLD_entities
    texts = ['Russ Cochran captured his first major title with his son as caddie.', 'Russ Cochran his reprints include EC Comics.', 'Russ Cochran has been publishing comic art.', "Russ Cochran was a member of University of Kentucky's golf team."]
    batch_deps_1 = [doc.to_array([ENT_KB_ID]) for doc in nlp.pipe(texts)]
    batch_deps_2 = [doc.to_array([ENT_KB_ID]) for doc in nlp.pipe(texts)]
    no_batch_deps = [doc.to_array([ENT_KB_ID]) for doc in [nlp(text) for text in texts]]
    assert_equal(batch_deps_1, batch_deps_2)
    assert_equal(batch_deps_1, no_batch_deps)

def test_kb_serialization():
    if False:
        return 10
    vector_length = 3
    with make_tempdir() as tmp_dir:
        kb_dir = tmp_dir / 'kb'
        nlp1 = English()
        assert 'Q2146908' not in nlp1.vocab.strings
        mykb = InMemoryLookupKB(nlp1.vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q2146908', freq=12, entity_vector=[6, -4, 3])
        mykb.add_alias(alias='Russ Cochran', entities=['Q2146908'], probabilities=[0.8])
        assert 'Q2146908' in nlp1.vocab.strings
        mykb.to_disk(kb_dir)
        nlp2 = English()
        assert 'RandomWord' not in nlp2.vocab.strings
        nlp2.vocab.strings.add('RandomWord')
        assert 'RandomWord' in nlp2.vocab.strings
        assert 'Q2146908' not in nlp2.vocab.strings
        entity_linker = nlp2.add_pipe('entity_linker', last=True)
        entity_linker.set_kb(load_kb(kb_dir))
        assert 'Q2146908' in nlp2.vocab.strings
        assert 'RandomWord' in nlp2.vocab.strings

@pytest.mark.xfail(reason='Needs fixing')
def test_kb_pickle():
    if False:
        while True:
            i = 10
    nlp = English()
    kb_1 = InMemoryLookupKB(nlp.vocab, entity_vector_length=3)
    kb_1.add_entity(entity='Q2146908', freq=12, entity_vector=[6, -4, 3])
    assert not kb_1.contains_alias('Russ Cochran')
    kb_1.add_alias(alias='Russ Cochran', entities=['Q2146908'], probabilities=[0.8])
    assert kb_1.contains_alias('Russ Cochran')
    data = pickle.dumps(kb_1)
    kb_2 = pickle.loads(data)
    assert kb_2.contains_alias('Russ Cochran')

@pytest.mark.xfail(reason='Needs fixing')
def test_nel_pickle():
    if False:
        print('Hello World!')

    def create_kb(vocab):
        if False:
            i = 10
            return i + 15
        kb = InMemoryLookupKB(vocab, entity_vector_length=3)
        kb.add_entity(entity='Q2146908', freq=12, entity_vector=[6, -4, 3])
        kb.add_alias(alias='Russ Cochran', entities=['Q2146908'], probabilities=[0.8])
        return kb
    nlp_1 = English()
    nlp_1.add_pipe('ner')
    entity_linker_1 = nlp_1.add_pipe('entity_linker', last=True)
    entity_linker_1.set_kb(create_kb)
    assert nlp_1.pipe_names == ['ner', 'entity_linker']
    assert entity_linker_1.kb.contains_alias('Russ Cochran')
    data = pickle.dumps(nlp_1)
    nlp_2 = pickle.loads(data)
    assert nlp_2.pipe_names == ['ner', 'entity_linker']
    entity_linker_2 = nlp_2.get_pipe('entity_linker')
    assert entity_linker_2.kb.contains_alias('Russ Cochran')

def test_kb_to_bytes():
    if False:
        while True:
            i = 10
    nlp = English()
    kb_1 = InMemoryLookupKB(nlp.vocab, entity_vector_length=3)
    kb_1.add_entity(entity='Q2146908', freq=12, entity_vector=[6, -4, 3])
    kb_1.add_entity(entity='Q66', freq=9, entity_vector=[1, 2, 3])
    kb_1.add_alias(alias='Russ Cochran', entities=['Q2146908'], probabilities=[0.8])
    kb_1.add_alias(alias='Boeing', entities=['Q66'], probabilities=[0.5])
    kb_1.add_alias(alias='Randomness', entities=['Q66', 'Q2146908'], probabilities=[0.1, 0.2])
    assert kb_1.contains_alias('Russ Cochran')
    kb_bytes = kb_1.to_bytes()
    kb_2 = InMemoryLookupKB(nlp.vocab, entity_vector_length=3)
    assert not kb_2.contains_alias('Russ Cochran')
    kb_2 = kb_2.from_bytes(kb_bytes)
    assert kb_1.get_size_entities() == kb_2.get_size_entities()
    assert kb_1.entity_vector_length == kb_2.entity_vector_length
    assert kb_1.get_entity_strings() == kb_2.get_entity_strings()
    assert kb_1.get_vector('Q2146908') == kb_2.get_vector('Q2146908')
    assert kb_1.get_vector('Q66') == kb_2.get_vector('Q66')
    assert kb_2.contains_alias('Russ Cochran')
    assert kb_1.get_size_aliases() == kb_2.get_size_aliases()
    assert kb_1.get_alias_strings() == kb_2.get_alias_strings()
    assert len(kb_1.get_alias_candidates('Russ Cochran')) == len(kb_2.get_alias_candidates('Russ Cochran'))
    assert len(kb_1.get_alias_candidates('Randomness')) == len(kb_2.get_alias_candidates('Randomness'))

def test_nel_to_bytes():
    if False:
        for i in range(10):
            print('nop')

    def create_kb(vocab):
        if False:
            print('Hello World!')
        kb = InMemoryLookupKB(vocab, entity_vector_length=3)
        kb.add_entity(entity='Q2146908', freq=12, entity_vector=[6, -4, 3])
        kb.add_alias(alias='Russ Cochran', entities=['Q2146908'], probabilities=[0.8])
        return kb
    nlp_1 = English()
    nlp_1.add_pipe('ner')
    entity_linker_1 = nlp_1.add_pipe('entity_linker', last=True)
    entity_linker_1.set_kb(create_kb)
    assert entity_linker_1.kb.contains_alias('Russ Cochran')
    assert nlp_1.pipe_names == ['ner', 'entity_linker']
    nlp_bytes = nlp_1.to_bytes()
    nlp_2 = English()
    nlp_2.add_pipe('ner')
    nlp_2.add_pipe('entity_linker', last=True)
    assert nlp_2.pipe_names == ['ner', 'entity_linker']
    assert not nlp_2.get_pipe('entity_linker').kb.contains_alias('Russ Cochran')
    nlp_2 = nlp_2.from_bytes(nlp_bytes)
    kb_2 = nlp_2.get_pipe('entity_linker').kb
    assert kb_2.contains_alias('Russ Cochran')
    assert kb_2.get_vector('Q2146908') == [6, -4, 3]
    assert_almost_equal(kb_2.get_prior_prob(entity='Q2146908', alias='Russ Cochran'), 0.8)

def test_scorer_links():
    if False:
        return 10
    train_examples = []
    nlp = English()
    ref1 = nlp('Julia lives in London happily.')
    ref1.ents = [Span(ref1, 0, 1, label='PERSON', kb_id='Q2'), Span(ref1, 3, 4, label='LOC', kb_id='Q3')]
    pred1 = nlp('Julia lives in London happily.')
    pred1.ents = [Span(pred1, 0, 1, label='PERSON', kb_id='Q70'), Span(pred1, 3, 4, label='LOC', kb_id='Q3')]
    train_examples.append(Example(pred1, ref1))
    ref2 = nlp('She loves London.')
    ref2.ents = [Span(ref2, 0, 1, label='PERSON', kb_id='Q2'), Span(ref2, 2, 3, label='LOC', kb_id='Q13')]
    pred2 = nlp('She loves London.')
    pred2.ents = [Span(pred2, 0, 1, label='PERSON', kb_id='Q2'), Span(pred2, 2, 3, label='LOC', kb_id='NIL')]
    train_examples.append(Example(pred2, ref2))
    ref3 = nlp('London is great.')
    ref3.ents = [Span(ref3, 0, 1, label='LOC', kb_id='NIL')]
    pred3 = nlp('London is great.')
    pred3.ents = [Span(pred3, 0, 1, label='LOC', kb_id='NIL')]
    train_examples.append(Example(pred3, ref3))
    scores = Scorer().score_links(train_examples, negative_labels=['NIL'])
    assert scores['nel_f_per_type']['PERSON']['p'] == 1 / 2
    assert scores['nel_f_per_type']['PERSON']['r'] == 1 / 2
    assert scores['nel_f_per_type']['LOC']['p'] == 1 / 1
    assert scores['nel_f_per_type']['LOC']['r'] == 1 / 2
    assert scores['nel_micro_p'] == 2 / 3
    assert scores['nel_micro_r'] == 2 / 4

@pytest.mark.parametrize('name,config', [('entity_linker', {'@architectures': 'spacy.EntityLinker.v1', 'tok2vec': DEFAULT_TOK2VEC_MODEL}), ('entity_linker', {'@architectures': 'spacy.EntityLinker.v2', 'tok2vec': DEFAULT_TOK2VEC_MODEL})])
def test_legacy_architectures(name, config):
    if False:
        while True:
            i = 10
    vector_length = 3
    nlp = English()
    train_examples = []
    for (text, annotation) in TRAIN_DATA:
        doc = nlp.make_doc(text)
        train_examples.append(Example.from_dict(doc, annotation))

    def create_kb(vocab):
        if False:
            while True:
                i = 10
        mykb = InMemoryLookupKB(vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q2146908', freq=12, entity_vector=[6, -4, 3])
        mykb.add_entity(entity='Q7381115', freq=12, entity_vector=[9, 1, -7])
        mykb.add_alias(alias='Russ Cochran', entities=['Q2146908', 'Q7381115'], probabilities=[0.5, 0.5])
        return mykb
    entity_linker = nlp.add_pipe(name, config={'model': config})
    if config['@architectures'] == 'spacy.EntityLinker.v1':
        assert isinstance(entity_linker, EntityLinker_v1)
    else:
        assert isinstance(entity_linker, EntityLinker)
    entity_linker.set_kb(create_kb)
    optimizer = nlp.initialize(get_examples=lambda : train_examples)
    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)

@pytest.mark.parametrize('patterns', [[{'label': 'CHARACTER', 'pattern': 'Kirby'}], [{'label': 'PERSON', 'pattern': 'Korby'}], [{'label': 'IS', 'pattern': 'is'}, {'label': 'COLOR', 'pattern': 'pink'}]])
def test_no_gold_ents(patterns):
    if False:
        for i in range(10):
            print('nop')
    TRAIN_DATA = [('Kirby is pink', {'links': {(0, 5): {'Q613241': 1.0}}, 'entities': [(0, 5, 'CHARACTER')], 'sent_starts': [1, 0, 0]})]
    nlp = English()
    vector_length = 3
    train_examples = []
    for (text, annotation) in TRAIN_DATA:
        doc = nlp(text)
        train_examples.append(Example.from_dict(doc, annotation))
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns(patterns)
    for eg in train_examples:
        eg.predicted = ruler(eg.predicted)
    nlp.remove_pipe('entity_ruler')

    def create_kb(vocab):
        if False:
            return 10
        mykb = InMemoryLookupKB(vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q613241', freq=12, entity_vector=[6, -4, 3])
        mykb.add_alias('Kirby', ['Q613241'], [0.9])
        mykb.add_entity(entity='pink', freq=12, entity_vector=[7, 2, -5])
        mykb.add_alias('pink', ['pink'], [0.9])
        return mykb
    entity_linker = nlp.add_pipe('entity_linker', config={'use_gold_ents': False}, last=True)
    entity_linker.set_kb(create_kb)
    assert entity_linker.use_gold_ents is False
    optimizer = nlp.initialize(get_examples=lambda : train_examples)
    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    nlp.add_pipe('sentencizer', first=True)
    nlp.evaluate(train_examples)

@pytest.mark.issue(9575)
def test_tokenization_mismatch():
    if False:
        i = 10
        return i + 15
    nlp = English()
    doc1 = Doc(nlp.vocab, words=['Kirby', '123456'], spaces=[True, False], ents=['B-CHARACTER', 'B-CARDINAL'])
    doc2 = Doc(nlp.vocab, words=['Kirby', '123', '456'], spaces=[True, False, False], ents=['B-CHARACTER', 'B-CARDINAL', 'B-CARDINAL'])
    eg = Example(doc1, doc2)
    train_examples = [eg]
    vector_length = 3

    def create_kb(vocab):
        if False:
            for i in range(10):
                print('nop')
        mykb = InMemoryLookupKB(vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q613241', freq=12, entity_vector=[6, -4, 3])
        mykb.add_alias('Kirby', ['Q613241'], [0.9])
        return mykb
    entity_linker = nlp.add_pipe('entity_linker', last=True)
    entity_linker.set_kb(create_kb)
    optimizer = nlp.initialize(get_examples=lambda : train_examples)
    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    nlp.add_pipe('sentencizer', first=True)
    nlp.evaluate(train_examples)

def test_abstract_kb_instantiation():
    if False:
        i = 10
        return i + 15
    'Test whether instantiation of abstract KB base class fails.'
    with pytest.raises(TypeError):
        KnowledgeBase(None, 3)

@pytest.mark.parametrize('meet_threshold,config', [(False, {'@architectures': 'spacy.EntityLinker.v2', 'tok2vec': DEFAULT_TOK2VEC_MODEL}), (True, {'@architectures': 'spacy.EntityLinker.v2', 'tok2vec': DEFAULT_TOK2VEC_MODEL})])
def test_threshold(meet_threshold: bool, config: Dict[str, Any]):
    if False:
        return 10
    'Tests abstention threshold.\n    meet_threshold (bool): Whether to configure NEL setup so that confidence threshold is met.\n    config (Dict[str, Any]): NEL architecture config.\n    '
    nlp = English()
    nlp.add_pipe('sentencizer')
    text = "Mahler's Symphony No. 8 was beautiful."
    entities = [(0, 6, 'PERSON')]
    links = {(0, 6): {'Q7304': 1.0}}
    sent_starts = [1, -1, 0, 0, 0, 0, 0, 0, 0]
    entity_id = 'Q7304'
    doc = nlp(text)
    train_examples = [Example.from_dict(doc, {'entities': entities, 'links': links, 'sent_starts': sent_starts})]

    def create_kb(vocab):
        if False:
            for i in range(10):
                print('nop')
        mykb = InMemoryLookupKB(vocab, entity_vector_length=3)
        mykb.add_entity(entity=entity_id, freq=12, entity_vector=[6, -4, 3])
        mykb.add_alias(alias='Mahler', entities=[entity_id], probabilities=[1 if meet_threshold else 0.01])
        return mykb
    entity_linker = nlp.add_pipe('entity_linker', last=True, config={'threshold': 0.99, 'model': config})
    entity_linker.set_kb(create_kb)
    nlp.initialize(get_examples=lambda : train_examples)
    ruler = nlp.add_pipe('entity_ruler', before='entity_linker')
    ruler.add_patterns([{'label': 'PERSON', 'pattern': [{'LOWER': 'mahler'}]}])
    doc = nlp(text)
    assert len(doc.ents) == 1
    assert doc.ents[0].kb_id_ == entity_id if meet_threshold else EntityLinker.NIL

def test_span_maker_forward_with_empty():
    if False:
        for i in range(10):
            print('nop')
    'The forward pass of the span maker may have a doc with no entities.'
    nlp = English()
    doc1 = nlp('a b c')
    ent = doc1[0:1]
    ent.label_ = 'X'
    doc1.ents = [ent]
    doc2 = nlp('x y z')
    span_maker = build_span_maker()
    span_maker([doc1, doc2], False)