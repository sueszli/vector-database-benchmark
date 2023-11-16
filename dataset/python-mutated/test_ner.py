import logging
import random
import pytest
from numpy.testing import assert_equal
from spacy import registry, util
from spacy.attrs import ENT_IOB
from spacy.lang.en import English
from spacy.lang.it import Italian
from spacy.language import Language
from spacy.lookups import Lookups
from spacy.pipeline import EntityRecognizer
from spacy.pipeline._parser_internals.ner import BiluoPushDown
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc, Span
from spacy.training import Example, iob_to_biluo, split_bilu_label
from spacy.vocab import Vocab
from ..util import make_tempdir
TRAIN_DATA = [('Who is Shaka Khan?', {'entities': [(7, 17, 'PERSON')]}), ('I like London and Berlin.', {'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]})]

@pytest.fixture
def neg_key():
    if False:
        for i in range(10):
            print('nop')
    return 'non_entities'

@pytest.fixture
def vocab():
    if False:
        print('Hello World!')
    return Vocab()

@pytest.fixture
def doc(vocab):
    if False:
        while True:
            i = 10
    return Doc(vocab, words=['Casey', 'went', 'to', 'New', 'York', '.'])

@pytest.fixture
def entity_annots(doc):
    if False:
        print('Hello World!')
    casey = doc[0:1]
    ny = doc[3:5]
    return [(casey.start_char, casey.end_char, 'PERSON'), (ny.start_char, ny.end_char, 'GPE')]

@pytest.fixture
def entity_types(entity_annots):
    if False:
        print('Hello World!')
    return sorted(set([label for (s, e, label) in entity_annots]))

@pytest.fixture
def tsys(vocab, entity_types):
    if False:
        while True:
            i = 10
    actions = BiluoPushDown.get_actions(entity_types=entity_types)
    return BiluoPushDown(vocab.strings, actions)

@pytest.mark.parametrize('label', ['U-JOB-NAME'])
@pytest.mark.issue(1967)
def test_issue1967(label):
    if False:
        i = 10
        return i + 15
    nlp = Language()
    config = {}
    ner = nlp.create_pipe('ner', config=config)
    example = Example.from_dict(Doc(ner.vocab, words=['word']), {'ids': [0], 'words': ['word'], 'tags': ['tag'], 'heads': [0], 'deps': ['dep'], 'entities': [label]})
    assert 'JOB-NAME' in ner.moves.get_actions(examples=[example])[1]

@pytest.mark.issue(2179)
def test_issue2179():
    if False:
        print('Hello World!')
    "Test that spurious 'extra_labels' aren't created when initializing NER."
    nlp = Italian()
    ner = nlp.add_pipe('ner')
    ner.add_label('CITIZENSHIP')
    nlp.initialize()
    nlp2 = Italian()
    nlp2.add_pipe('ner')
    assert len(nlp2.get_pipe('ner').labels) == 0
    model = nlp2.get_pipe('ner').model
    model.attrs['resize_output'](model, nlp.get_pipe('ner').moves.n_moves)
    nlp2.from_bytes(nlp.to_bytes())
    assert 'extra_labels' not in nlp2.get_pipe('ner').cfg
    assert nlp2.get_pipe('ner').labels == ('CITIZENSHIP',)

@pytest.mark.issue(2385)
def test_issue2385():
    if False:
        i = 10
        return i + 15
    'Test that IOB tags are correctly converted to BILUO tags.'
    tags1 = ('B-BRAWLER', 'I-BRAWLER', 'I-BRAWLER')
    assert iob_to_biluo(tags1) == ['B-BRAWLER', 'I-BRAWLER', 'L-BRAWLER']
    tags2 = ('I-ORG', 'I-ORG', 'B-ORG')
    assert iob_to_biluo(tags2) == ['B-ORG', 'L-ORG', 'U-ORG']
    tags3 = ('B-PERSON', 'I-PERSON', 'B-PERSON')
    assert iob_to_biluo(tags3) == ['B-PERSON', 'L-PERSON', 'U-PERSON']
    tags4 = ('B-MULTI-PERSON', 'I-MULTI-PERSON', 'B-MULTI-PERSON')
    assert iob_to_biluo(tags4) == ['B-MULTI-PERSON', 'L-MULTI-PERSON', 'U-MULTI-PERSON']

@pytest.mark.issue(2800)
def test_issue2800():
    if False:
        return 10
    'Test issue that arises when too many labels are added to NER model.\n    Used to cause segfault.\n    '
    nlp = English()
    train_data = []
    train_data.extend([Example.from_dict(nlp.make_doc('One sentence'), {'entities': []})])
    entity_types = [str(i) for i in range(1000)]
    ner = nlp.add_pipe('ner')
    for entity_type in list(entity_types):
        ner.add_label(entity_type)
    optimizer = nlp.initialize()
    for i in range(20):
        losses = {}
        random.shuffle(train_data)
        for example in train_data:
            nlp.update([example], sgd=optimizer, losses=losses, drop=0.5)

@pytest.mark.issue(3209)
def test_issue3209():
    if False:
        i = 10
        return i + 15
    'Test issue that occurred in spaCy nightly where NER labels were being\n    mapped to classes incorrectly after loading the model, when the labels\n    were added using ner.add_label().\n    '
    nlp = English()
    ner = nlp.add_pipe('ner')
    ner.add_label('ANIMAL')
    nlp.initialize()
    move_names = ['O', 'B-ANIMAL', 'I-ANIMAL', 'L-ANIMAL', 'U-ANIMAL']
    assert ner.move_names == move_names
    nlp2 = English()
    ner2 = nlp2.add_pipe('ner')
    model = ner2.model
    model.attrs['resize_output'](model, ner.moves.n_moves)
    nlp2.from_bytes(nlp.to_bytes())
    assert ner2.move_names == move_names

def test_labels_from_BILUO():
    if False:
        print('Hello World!')
    "Test that labels are inferred correctly when there's a - in label."
    nlp = English()
    ner = nlp.add_pipe('ner')
    ner.add_label('LARGE-ANIMAL')
    nlp.initialize()
    move_names = ['O', 'B-LARGE-ANIMAL', 'I-LARGE-ANIMAL', 'L-LARGE-ANIMAL', 'U-LARGE-ANIMAL']
    labels = {'LARGE-ANIMAL'}
    assert ner.move_names == move_names
    assert set(ner.labels) == labels

@pytest.mark.issue(4267)
def test_issue4267():
    if False:
        i = 10
        return i + 15
    'Test that running an entity_ruler after ner gives consistent results'
    nlp = English()
    ner = nlp.add_pipe('ner')
    ner.add_label('PEOPLE')
    nlp.initialize()
    assert 'ner' in nlp.pipe_names
    doc1 = nlp('hi')
    assert doc1.has_annotation('ENT_IOB')
    for token in doc1:
        assert token.ent_iob == 2
    patterns = [{'label': 'SOFTWARE', 'pattern': 'spacy'}]
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns(patterns)
    assert 'entity_ruler' in nlp.pipe_names
    assert 'ner' in nlp.pipe_names
    doc2 = nlp('hi')
    assert doc2.has_annotation('ENT_IOB')
    for token in doc2:
        assert token.ent_iob == 2

@pytest.mark.issue(4313)
def test_issue4313():
    if False:
        return 10
    'This should not crash or exit with some strange error code'
    beam_width = 16
    beam_density = 0.0001
    nlp = English()
    config = {'beam_width': beam_width, 'beam_density': beam_density}
    ner = nlp.add_pipe('beam_ner', config=config)
    ner.add_label('SOME_LABEL')
    nlp.initialize()
    doc = nlp('What do you think about Apple ?')
    assert len(ner.labels) == 1
    assert 'SOME_LABEL' in ner.labels
    apple_ent = Span(doc, 5, 6, label='MY_ORG')
    doc.ents = list(doc.ents) + [apple_ent]
    docs = [doc]
    ner.beam_parse(docs, drop=0.0, beam_width=beam_width, beam_density=beam_density)
    assert len(ner.labels) == 2
    assert 'MY_ORG' in ner.labels

def test_get_oracle_moves(tsys, doc, entity_annots):
    if False:
        print('Hello World!')
    example = Example.from_dict(doc, {'entities': entity_annots})
    act_classes = tsys.get_oracle_sequence(example, _debug=False)
    names = [tsys.get_class_name(act) for act in act_classes]
    assert names == ['U-PERSON', 'O', 'O', 'B-GPE', 'L-GPE', 'O']

def test_negative_samples_two_word_input(tsys, vocab, neg_key):
    if False:
        return 10
    "Test that we don't get stuck in a two word input when we have a negative\n    span. This could happen if we don't have the right check on the B action.\n    "
    tsys.cfg['neg_key'] = neg_key
    doc = Doc(vocab, words=['A', 'B'])
    entity_annots = [None, None]
    example = Example.from_dict(doc, {'entities': entity_annots})
    example.y.spans[neg_key] = [Span(example.y, 0, 1, label='O'), Span(example.y, 0, 2, label='PERSON')]
    act_classes = tsys.get_oracle_sequence(example)
    names = [tsys.get_class_name(act) for act in act_classes]
    assert names
    assert names[0] != 'O'
    assert names[0] != 'B-PERSON'
    assert names[1] != 'L-PERSON'

def test_negative_samples_three_word_input(tsys, vocab, neg_key):
    if False:
        i = 10
        return i + 15
    'Test that we exclude a 2-word entity correctly using a negative example.'
    tsys.cfg['neg_key'] = neg_key
    doc = Doc(vocab, words=['A', 'B', 'C'])
    entity_annots = [None, None, None]
    example = Example.from_dict(doc, {'entities': entity_annots})
    example.y.spans[neg_key] = [Span(example.y, 0, 1, label='O'), Span(example.y, 0, 2, label='PERSON')]
    act_classes = tsys.get_oracle_sequence(example)
    names = [tsys.get_class_name(act) for act in act_classes]
    assert names
    assert names[0] != 'O'
    assert names[1] != 'B-PERSON'

def test_negative_samples_U_entity(tsys, vocab, neg_key):
    if False:
        while True:
            i = 10
    'Test that we exclude a 2-word entity correctly using a negative example.'
    tsys.cfg['neg_key'] = neg_key
    doc = Doc(vocab, words=['A'])
    entity_annots = [None]
    example = Example.from_dict(doc, {'entities': entity_annots})
    example.y.spans[neg_key] = [Span(example.y, 0, 1, label='O'), Span(example.y, 0, 1, label='PERSON')]
    act_classes = tsys.get_oracle_sequence(example)
    names = [tsys.get_class_name(act) for act in act_classes]
    assert names
    assert names[0] != 'O'
    assert names[0] != 'U-PERSON'

def test_negative_sample_key_is_in_config(vocab, entity_types):
    if False:
        i = 10
        return i + 15
    actions = BiluoPushDown.get_actions(entity_types=entity_types)
    tsys = BiluoPushDown(vocab.strings, actions, incorrect_spans_key='non_entities')
    assert tsys.cfg['neg_key'] == 'non_entities'

@pytest.mark.skip(reason='No longer supported')
def test_oracle_moves_missing_B(en_vocab):
    if False:
        return 10
    words = ['B', '52', 'Bomber']
    biluo_tags = [None, None, 'L-PRODUCT']
    doc = Doc(en_vocab, words=words)
    example = Example.from_dict(doc, {'words': words, 'entities': biluo_tags})
    moves = BiluoPushDown(en_vocab.strings)
    move_types = ('M', 'B', 'I', 'L', 'U', 'O')
    for tag in biluo_tags:
        if tag is None:
            continue
        elif tag == 'O':
            moves.add_action(move_types.index('O'), '')
        else:
            (action, label) = split_bilu_label(tag)
            moves.add_action(move_types.index('B'), label)
            moves.add_action(move_types.index('I'), label)
            moves.add_action(move_types.index('L'), label)
            moves.add_action(move_types.index('U'), label)
    moves.get_oracle_sequence(example)

@pytest.mark.skip(reason='No longer supported')
def test_oracle_moves_whitespace(en_vocab):
    if False:
        while True:
            i = 10
    words = ['production', '\n', 'of', 'Northrop', '\n', 'Corp.', '\n', "'s", 'radar']
    biluo_tags = ['O', 'O', 'O', 'B-ORG', None, 'I-ORG', 'L-ORG', 'O', 'O']
    doc = Doc(en_vocab, words=words)
    example = Example.from_dict(doc, {'entities': biluo_tags})
    moves = BiluoPushDown(en_vocab.strings)
    move_types = ('M', 'B', 'I', 'L', 'U', 'O')
    for tag in biluo_tags:
        if tag is None:
            continue
        elif tag == 'O':
            moves.add_action(move_types.index('O'), '')
        else:
            (action, label) = split_bilu_label(tag)
            moves.add_action(move_types.index(action), label)
    moves.get_oracle_sequence(example)

def test_accept_blocked_token():
    if False:
        print('Hello World!')
    'Test succesful blocking of tokens to be in an entity.'
    nlp1 = English()
    doc1 = nlp1('I live in New York')
    config = {}
    ner1 = nlp1.create_pipe('ner', config=config)
    assert [token.ent_iob_ for token in doc1] == ['', '', '', '', '']
    assert [token.ent_type_ for token in doc1] == ['', '', '', '', '']
    ner1.moves.add_action(5, '')
    ner1.add_label('GPE')
    state1 = ner1.moves.init_batch([doc1])[0]
    ner1.moves.apply_transition(state1, 'O')
    ner1.moves.apply_transition(state1, 'O')
    ner1.moves.apply_transition(state1, 'O')
    assert ner1.moves.is_valid(state1, 'B-GPE')
    nlp2 = English()
    doc2 = nlp2('I live in New York')
    config = {}
    ner2 = nlp2.create_pipe('ner', config=config)
    doc2.set_ents([], blocked=[doc2[3:5]], default='unmodified')
    assert [token.ent_iob_ for token in doc2] == ['', '', '', 'B', 'B']
    assert [token.ent_type_ for token in doc2] == ['', '', '', '', '']
    ner2.moves.add_action(4, '')
    ner2.moves.add_action(5, '')
    ner2.add_label('GPE')
    state2 = ner2.moves.init_batch([doc2])[0]
    ner2.moves.apply_transition(state2, 'O')
    ner2.moves.apply_transition(state2, 'O')
    ner2.moves.apply_transition(state2, 'O')
    assert not ner2.moves.is_valid(state2, 'B-GPE')
    assert ner2.moves.is_valid(state2, 'U-')
    ner2.moves.apply_transition(state2, 'U-')
    assert not ner2.moves.is_valid(state2, 'B-GPE')
    assert ner2.moves.is_valid(state2, 'U-')

def test_train_empty():
    if False:
        return 10
    'Test that training an empty text does not throw errors.'
    train_data = [('Who is Shaka Khan?', {'entities': [(7, 17, 'PERSON')]}), ('', {'entities': []})]
    nlp = English()
    train_examples = []
    for t in train_data:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    ner = nlp.add_pipe('ner', last=True)
    ner.add_label('PERSON')
    nlp.initialize()
    for itn in range(2):
        losses = {}
        batches = util.minibatch(train_examples, size=8)
        for batch in batches:
            nlp.update(batch, losses=losses)

def test_train_negative_deprecated():
    if False:
        i = 10
        return i + 15
    'Test that the deprecated negative entity format raises a custom error.'
    train_data = [('Who is Shaka Khan?', {'entities': [(7, 17, '!PERSON')]})]
    nlp = English()
    train_examples = []
    for t in train_data:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    ner = nlp.add_pipe('ner', last=True)
    ner.add_label('PERSON')
    nlp.initialize()
    for itn in range(2):
        losses = {}
        batches = util.minibatch(train_examples, size=8)
        for batch in batches:
            with pytest.raises(ValueError):
                nlp.update(batch, losses=losses)

def test_overwrite_token():
    if False:
        while True:
            i = 10
    nlp = English()
    nlp.add_pipe('ner')
    nlp.initialize()
    doc = nlp('I live in New York')
    assert [token.ent_iob_ for token in doc] == ['O', 'O', 'O', 'O', 'O']
    assert [token.ent_type_ for token in doc] == ['', '', '', '', '']
    config = {}
    ner2 = nlp.create_pipe('ner', config=config)
    ner2.moves.add_action(5, '')
    ner2.add_label('GPE')
    state = ner2.moves.init_batch([doc])[0]
    assert ner2.moves.is_valid(state, 'B-GPE')
    assert ner2.moves.is_valid(state, 'U-GPE')
    ner2.moves.apply_transition(state, 'B-GPE')
    assert ner2.moves.is_valid(state, 'I-GPE')
    assert ner2.moves.is_valid(state, 'L-GPE')

def test_empty_ner():
    if False:
        while True:
            i = 10
    nlp = English()
    ner = nlp.add_pipe('ner')
    ner.add_label('MY_LABEL')
    nlp.initialize()
    doc = nlp("John is watching the news about Croatia's elections")
    result = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    assert [token.ent_iob_ for token in doc] == result

def test_ruler_before_ner():
    if False:
        while True:
            i = 10
    'Test that an NER works after an entity_ruler: the second can add annotations'
    nlp = English()
    patterns = [{'label': 'THING', 'pattern': 'This'}]
    ruler = nlp.add_pipe('entity_ruler')
    untrained_ner = nlp.add_pipe('ner')
    untrained_ner.add_label('MY_LABEL')
    nlp.initialize()
    ruler.add_patterns(patterns)
    doc = nlp('This is Antti Korhonen speaking in Finland')
    expected_iobs = ['B', 'O', 'O', 'O', 'O', 'O', 'O']
    expected_types = ['THING', '', '', '', '', '', '']
    assert [token.ent_iob_ for token in doc] == expected_iobs
    assert [token.ent_type_ for token in doc] == expected_types

def test_ner_constructor(en_vocab):
    if False:
        i = 10
        return i + 15
    config = {'update_with_oracle_cut_size': 100}
    cfg = {'model': DEFAULT_NER_MODEL}
    model = registry.resolve(cfg, validate=True)['model']
    EntityRecognizer(en_vocab, model, **config)
    EntityRecognizer(en_vocab, model)

def test_ner_before_ruler():
    if False:
        while True:
            i = 10
    'Test that an entity_ruler works after an NER: the second can overwrite O annotations'
    nlp = English()
    untrained_ner = nlp.add_pipe('ner', name='uner')
    untrained_ner.add_label('MY_LABEL')
    nlp.initialize()
    patterns = [{'label': 'THING', 'pattern': 'This'}]
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns(patterns)
    doc = nlp('This is Antti Korhonen speaking in Finland')
    expected_iobs = ['B', 'O', 'O', 'O', 'O', 'O', 'O']
    expected_types = ['THING', '', '', '', '', '', '']
    assert [token.ent_iob_ for token in doc] == expected_iobs
    assert [token.ent_type_ for token in doc] == expected_types

def test_block_ner():
    if False:
        i = 10
        return i + 15
    "Test functionality for blocking tokens so they can't be in a named entity"
    nlp = English()
    nlp.add_pipe('blocker', config={'start': 2, 'end': 5})
    untrained_ner = nlp.add_pipe('ner')
    untrained_ner.add_label('MY_LABEL')
    nlp.initialize()
    doc = nlp('This is Antti L Korhonen speaking in Finland')
    expected_iobs = ['O', 'O', 'B', 'B', 'B', 'O', 'O', 'O']
    expected_types = ['', '', '', '', '', '', '', '']
    assert [token.ent_iob_ for token in doc] == expected_iobs
    assert [token.ent_type_ for token in doc] == expected_types

@pytest.mark.parametrize('use_upper', [True, False])
def test_overfitting_IO(use_upper):
    if False:
        return 10
    nlp = English()
    ner = nlp.add_pipe('ner', config={'model': {'use_upper': use_upper}})
    train_examples = []
    for (text, annotations) in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    optimizer = nlp.initialize()
    for i in range(50):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses['ner'] < 1e-05
    test_text = 'I like London.'
    doc = nlp(test_text)
    ents = doc.ents
    assert len(ents) == 1
    assert ents[0].text == 'London'
    assert ents[0].label_ == 'LOC'
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        doc2 = nlp2(test_text)
        ents2 = doc2.ents
        assert len(ents2) == 1
        assert ents2[0].text == 'London'
        assert ents2[0].label_ == 'LOC'
        ner2 = nlp2.get_pipe('ner')
        assert ner2.model.attrs['has_upper'] == use_upper
        ner2.add_label('RANDOM_NEW_LABEL')
        doc3 = nlp2(test_text)
        ents3 = doc3.ents
        assert len(ents3) == 1
        assert ents3[0].text == 'London'
        assert ents3[0].label_ == 'LOC'
    texts = ['Just a sentence.', 'Then one more sentence about London.', 'Here is another one.', 'I like London.']
    batch_deps_1 = [doc.to_array([ENT_IOB]) for doc in nlp.pipe(texts)]
    batch_deps_2 = [doc.to_array([ENT_IOB]) for doc in nlp.pipe(texts)]
    no_batch_deps = [doc.to_array([ENT_IOB]) for doc in [nlp(text) for text in texts]]
    assert_equal(batch_deps_1, batch_deps_2)
    assert_equal(batch_deps_1, no_batch_deps)
    test_text = 'I like London and London.'
    doc = nlp.make_doc(test_text)
    doc.ents = [Span(doc, 2, 3, label='LOC', kb_id=1234)]
    ents = doc.ents
    assert len(ents) == 1
    assert ents[0].text == 'London'
    assert ents[0].label_ == 'LOC'
    assert ents[0].kb_id == 1234
    doc = nlp.get_pipe('ner')(doc)
    ents = doc.ents
    assert len(ents) == 2
    assert ents[0].text == 'London'
    assert ents[0].label_ == 'LOC'
    assert ents[0].kb_id == 1234
    assert ents[1].text == 'London'
    assert ents[1].label_ == 'LOC'
    assert ents[1].kb_id == 0

def test_beam_ner_scores():
    if False:
        for i in range(10):
            print('nop')
    beam_width = 16
    beam_density = 0.0001
    nlp = English()
    config = {'beam_width': beam_width, 'beam_density': beam_density}
    ner = nlp.add_pipe('beam_ner', config=config)
    train_examples = []
    for (text, annotations) in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    optimizer = nlp.initialize()
    losses = {}
    nlp.update(train_examples, sgd=optimizer, losses=losses)
    test_text = 'I like London.'
    doc = nlp.make_doc(test_text)
    docs = [doc]
    beams = ner.predict(docs)
    entity_scores = ner.scored_ents(beams)[0]
    for j in range(len(doc)):
        for label in ner.labels:
            score = entity_scores[j, j + 1, label]
            eps = 1e-05
            assert 0 - eps <= score <= 1 + eps

def test_beam_overfitting_IO(neg_key):
    if False:
        print('Hello World!')
    nlp = English()
    beam_width = 16
    beam_density = 0.0001
    config = {'beam_width': beam_width, 'beam_density': beam_density, 'incorrect_spans_key': neg_key}
    ner = nlp.add_pipe('beam_ner', config=config)
    train_examples = []
    for (text, annotations) in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    optimizer = nlp.initialize()
    for i in range(50):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    assert losses['beam_ner'] < 0.0001
    test_text = 'I like London'
    docs = [nlp.make_doc(test_text)]
    beams = ner.predict(docs)
    entity_scores = ner.scored_ents(beams)[0]
    assert entity_scores[2, 3, 'LOC'] == 1.0
    assert entity_scores[2, 3, 'PERSON'] == 0.0
    assert len(nlp(test_text).ents) == 1
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        nlp2 = util.load_model_from_path(tmp_dir)
        docs2 = [nlp2.make_doc(test_text)]
        ner2 = nlp2.get_pipe('beam_ner')
        beams2 = ner2.predict(docs2)
        entity_scores2 = ner2.scored_ents(beams2)[0]
        assert entity_scores2[2, 3, 'LOC'] == 1.0
        assert entity_scores2[2, 3, 'PERSON'] == 0.0
    neg_doc = nlp.make_doc(test_text)
    neg_ex = Example(neg_doc, neg_doc)
    neg_ex.reference.spans[neg_key] = [Span(neg_doc, 2, 3, 'LOC')]
    neg_train_examples = [neg_ex]
    for i in range(20):
        losses = {}
        nlp.update(neg_train_examples, sgd=optimizer, losses=losses)
    assert len(nlp(test_text).ents) == 0

def test_neg_annotation(neg_key):
    if False:
        return 10
    'Check that the NER update works with a negative annotation that is a different label of the correct one,\n    or partly overlapping, etc'
    nlp = English()
    beam_width = 16
    beam_density = 0.0001
    config = {'beam_width': beam_width, 'beam_density': beam_density, 'incorrect_spans_key': neg_key}
    ner = nlp.add_pipe('beam_ner', config=config)
    train_text = 'Who is Shaka Khan?'
    neg_doc = nlp.make_doc(train_text)
    ner.add_label('PERSON')
    ner.add_label('ORG')
    example = Example.from_dict(neg_doc, {'entities': [(7, 17, 'PERSON')]})
    example.reference.spans[neg_key] = [Span(example.reference, 2, 4, 'ORG'), Span(example.reference, 2, 3, 'PERSON'), Span(example.reference, 1, 4, 'PERSON')]
    optimizer = nlp.initialize()
    for i in range(2):
        losses = {}
        nlp.update([example], sgd=optimizer, losses=losses)

def test_neg_annotation_conflict(neg_key):
    if False:
        for i in range(10):
            print('nop')
    nlp = English()
    beam_width = 16
    beam_density = 0.0001
    config = {'beam_width': beam_width, 'beam_density': beam_density, 'incorrect_spans_key': neg_key}
    ner = nlp.add_pipe('beam_ner', config=config)
    train_text = 'Who is Shaka Khan?'
    neg_doc = nlp.make_doc(train_text)
    ner.add_label('PERSON')
    ner.add_label('LOC')
    example = Example.from_dict(neg_doc, {'entities': [(7, 17, 'PERSON')]})
    example.reference.spans[neg_key] = [Span(example.reference, 2, 4, 'PERSON')]
    assert len(example.reference.ents) == 1
    assert example.reference.ents[0].text == 'Shaka Khan'
    assert example.reference.ents[0].label_ == 'PERSON'
    assert len(example.reference.spans[neg_key]) == 1
    assert example.reference.spans[neg_key][0].text == 'Shaka Khan'
    assert example.reference.spans[neg_key][0].label_ == 'PERSON'
    optimizer = nlp.initialize()
    for i in range(2):
        losses = {}
        with pytest.raises(ValueError):
            nlp.update([example], sgd=optimizer, losses=losses)

def test_beam_valid_parse(neg_key):
    if False:
        while True:
            i = 10
    'Regression test for previously flakey behaviour'
    nlp = English()
    beam_width = 16
    beam_density = 0.0001
    config = {'beam_width': beam_width, 'beam_density': beam_density, 'incorrect_spans_key': neg_key}
    nlp.add_pipe('beam_ner', config=config)
    tokens = ['FEDERAL', 'NATIONAL', 'MORTGAGE', 'ASSOCIATION', '(', 'Fannie', 'Mae', '):', 'Posted', 'yields', 'on', '30', 'year', 'mortgage', 'commitments', 'for', 'delivery', 'within', '30', 'days', '(', 'priced', 'at', 'par', ')', '9.75', '%', ',', 'standard', 'conventional', 'fixed', '-', 'rate', 'mortgages', ';', '8.70', '%', ',', '6/2', 'rate', 'capped', 'one', '-', 'year', 'adjustable', 'rate', 'mortgages', '.', 'Source', ':', 'Telerate', 'Systems', 'Inc.']
    iob = ['B-ORG', 'I-ORG', 'I-ORG', 'L-ORG', 'O', 'B-ORG', 'L-ORG', 'O', 'O', 'O', 'O', 'B-DATE', 'L-DATE', 'O', 'O', 'O', 'O', 'O', 'B-DATE', 'L-DATE', 'O', 'O', 'O', 'O', 'O', 'B-PERCENT', 'L-PERCENT', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PERCENT', 'L-PERCENT', 'O', 'U-CARDINAL', 'O', 'O', 'B-DATE', 'I-DATE', 'L-DATE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    doc = Doc(nlp.vocab, words=tokens)
    example = Example.from_dict(doc, {'ner': iob})
    neg_span = Span(example.reference, 50, 53, 'ORG')
    example.reference.spans[neg_key] = [neg_span]
    optimizer = nlp.initialize()
    for i in range(5):
        losses = {}
        nlp.update([example], sgd=optimizer, losses=losses)
    assert 'beam_ner' in losses

def test_ner_warns_no_lookups(caplog):
    if False:
        for i in range(10):
            print('nop')
    nlp = English()
    assert nlp.lang in util.LEXEME_NORM_LANGS
    nlp.vocab.lookups = Lookups()
    assert not len(nlp.vocab.lookups)
    nlp.add_pipe('ner')
    with caplog.at_level(logging.DEBUG):
        nlp.initialize()
        assert 'W033' in caplog.text
    caplog.clear()
    nlp.vocab.lookups.add_table('lexeme_norm')
    nlp.vocab.lookups.get_table('lexeme_norm')['a'] = 'A'
    with caplog.at_level(logging.DEBUG):
        nlp.initialize()
        assert 'W033' not in caplog.text

@Language.factory('blocker')
class BlockerComponent1:

    def __init__(self, nlp, start, end, name='my_blocker'):
        if False:
            print('Hello World!')
        self.start = start
        self.end = end
        self.name = name

    def __call__(self, doc):
        if False:
            while True:
                i = 10
        doc.set_ents([], blocked=[doc[self.start:self.end]], default='unmodified')
        return doc