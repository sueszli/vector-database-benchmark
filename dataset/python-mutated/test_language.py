import itertools
import logging
from unittest import mock
import pytest
from thinc.api import CupyOps, NumpyOps, get_current_ops
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import find_matching_language, ignore_error, raise_error, registry
from spacy.vocab import Vocab
from .util import add_vecs_to_vocab, assert_docs_equal
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except ImportError:
    pass

def evil_component(doc):
    if False:
        for i in range(10):
            print('nop')
    if '2' in doc.text:
        raise ValueError('no dice')
    return doc

def perhaps_set_sentences(doc):
    if False:
        for i in range(10):
            print('nop')
    if not doc.text.startswith('4'):
        doc[-1].is_sent_start = True
    return doc

def assert_sents_error(doc):
    if False:
        for i in range(10):
            print('nop')
    if not doc.has_annotation('SENT_START'):
        raise ValueError('no sents')
    return doc

def warn_error(proc_name, proc, docs, e):
    if False:
        for i in range(10):
            print('nop')
    logger = logging.getLogger('spacy')
    logger.warning('Trouble with component %s.', proc_name)

@pytest.fixture
def nlp():
    if False:
        while True:
            i = 10
    nlp = Language(Vocab())
    textcat = nlp.add_pipe('textcat')
    for label in ('POSITIVE', 'NEGATIVE'):
        textcat.add_label(label)
    nlp.initialize()
    return nlp

def test_language_update(nlp):
    if False:
        i = 10
        return i + 15
    text = 'hello world'
    annots = {'cats': {'POSITIVE': 1.0, 'NEGATIVE': 0.0}}
    wrongkeyannots = {'LABEL': True}
    doc = Doc(nlp.vocab, words=text.split(' '))
    example = Example.from_dict(doc, annots)
    nlp.update([example])
    with pytest.raises(TypeError):
        nlp.update(example)
    with pytest.raises(TypeError):
        nlp.update((text, annots))
    with pytest.raises(TypeError):
        nlp.update((doc, annots))
    with pytest.raises(ValueError):
        example = Example.from_dict(doc, None)
    with pytest.raises(KeyError):
        example = Example.from_dict(doc, wrongkeyannots)

def test_language_evaluate(nlp):
    if False:
        return 10
    text = 'hello world'
    annots = {'doc_annotation': {'cats': {'POSITIVE': 1.0, 'NEGATIVE': 0.0}}}
    doc = Doc(nlp.vocab, words=text.split(' '))
    example = Example.from_dict(doc, annots)
    scores = nlp.evaluate([example])
    assert scores['speed'] > 0
    scores = nlp.evaluate((eg for eg in [example]))
    assert scores['speed'] > 0
    with pytest.raises(TypeError):
        nlp.evaluate(example)
    with pytest.raises(TypeError):
        nlp.evaluate([(text, annots)])
    with pytest.raises(TypeError):
        nlp.evaluate([(doc, annots)])
    with pytest.raises(TypeError):
        nlp.evaluate([text, annots])

def test_evaluate_no_pipe(nlp):
    if False:
        print('Hello World!')
    "Test that docs are processed correctly within Language.pipe if the\n    component doesn't expose a .pipe method."

    @Language.component('test_evaluate_no_pipe')
    def pipe(doc):
        if False:
            i = 10
            return i + 15
        return doc
    text = 'hello world'
    annots = {'cats': {'POSITIVE': 1.0, 'NEGATIVE': 0.0}}
    nlp = Language(Vocab())
    doc = nlp(text)
    nlp.add_pipe('test_evaluate_no_pipe')
    nlp.evaluate([Example.from_dict(doc, annots)])

def test_evaluate_textcat_multilabel(en_vocab):
    if False:
        while True:
            i = 10
    'Test that evaluate works with a multilabel textcat pipe.'
    nlp = Language(en_vocab)
    textcat_multilabel = nlp.add_pipe('textcat_multilabel')
    for label in ('FEATURE', 'REQUEST', 'BUG', 'QUESTION'):
        textcat_multilabel.add_label(label)
    nlp.initialize()
    annots = {'cats': {'FEATURE': 1.0, 'QUESTION': 1.0}}
    doc = nlp.make_doc('hello world')
    example = Example.from_dict(doc, annots)
    scores = nlp.evaluate([example])
    labels = nlp.get_pipe('textcat_multilabel').labels
    for label in labels:
        assert scores['cats_f_per_type'].get(label) is not None
    for key in example.reference.cats.keys():
        if key not in labels:
            assert scores['cats_f_per_type'].get(key) is None

def test_evaluate_multiple_textcat_final(en_vocab):
    if False:
        i = 10
        return i + 15
    'Test that evaluate evaluates the final textcat component in a pipeline\n    with more than one textcat or textcat_multilabel.'
    nlp = Language(en_vocab)
    textcat = nlp.add_pipe('textcat')
    for label in ('POSITIVE', 'NEGATIVE'):
        textcat.add_label(label)
    textcat_multilabel = nlp.add_pipe('textcat_multilabel')
    for label in ('FEATURE', 'REQUEST', 'BUG', 'QUESTION'):
        textcat_multilabel.add_label(label)
    nlp.initialize()
    annots = {'cats': {'POSITIVE': 1.0, 'NEGATIVE': 0.0, 'FEATURE': 1.0, 'QUESTION': 1.0, 'POSITIVE': 1.0, 'NEGATIVE': 0.0}}
    doc = nlp.make_doc('hello world')
    example = Example.from_dict(doc, annots)
    scores = nlp.evaluate([example])
    labels = nlp.get_pipe(nlp.pipe_names[-1]).labels
    for label in labels:
        assert scores['cats_f_per_type'].get(label) is not None
    for key in example.reference.cats.keys():
        if key not in labels:
            assert scores['cats_f_per_type'].get(key) is None

def test_evaluate_multiple_textcat_separate(en_vocab):
    if False:
        for i in range(10):
            print('nop')
    'Test that evaluate can evaluate multiple textcat components separately\n    with custom scorers.'

    def custom_textcat_score(examples, **kwargs):
        if False:
            print('Hello World!')
        scores = Scorer.score_cats(examples, 'cats', multi_label=False, **kwargs)
        return {f'custom_{k}': v for (k, v) in scores.items()}

    @spacy.registry.scorers('test_custom_textcat_scorer')
    def make_custom_textcat_scorer():
        if False:
            while True:
                i = 10
        return custom_textcat_score
    nlp = Language(en_vocab)
    textcat = nlp.add_pipe('textcat', config={'scorer': {'@scorers': 'test_custom_textcat_scorer'}})
    for label in ('POSITIVE', 'NEGATIVE'):
        textcat.add_label(label)
    textcat_multilabel = nlp.add_pipe('textcat_multilabel')
    for label in ('FEATURE', 'REQUEST', 'BUG', 'QUESTION'):
        textcat_multilabel.add_label(label)
    nlp.initialize()
    annots = {'cats': {'POSITIVE': 1.0, 'NEGATIVE': 0.0, 'FEATURE': 1.0, 'QUESTION': 1.0, 'POSITIVE': 1.0, 'NEGATIVE': 0.0}}
    doc = nlp.make_doc('hello world')
    example = Example.from_dict(doc, annots)
    scores = nlp.evaluate([example])
    assert 'custom_cats_f_per_type' in scores
    labels = nlp.get_pipe('textcat').labels
    assert set(scores['custom_cats_f_per_type'].keys()) == set(labels)
    assert 'cats_f_per_type' in scores
    labels = nlp.get_pipe('textcat_multilabel').labels
    assert set(scores['cats_f_per_type'].keys()) == set(labels)

def vector_modification_pipe(doc):
    if False:
        return 10
    doc.vector += 1
    return doc

def userdata_pipe(doc):
    if False:
        for i in range(10):
            print('nop')
    doc.user_data['foo'] = 'bar'
    return doc

def ner_pipe(doc):
    if False:
        i = 10
        return i + 15
    span = Span(doc, 0, 1, label='FIRST')
    doc.ents += (span,)
    return doc

@pytest.fixture
def sample_vectors():
    if False:
        print('Hello World!')
    return [('spacy', [-0.1, -0.2, -0.3]), ('world', [-0.2, -0.3, -0.4]), ('pipe', [0.7, 0.8, 0.9])]

@pytest.fixture
def nlp2(nlp, sample_vectors):
    if False:
        print('Hello World!')
    Language.component('test_language_vector_modification_pipe', func=vector_modification_pipe)
    Language.component('test_language_userdata_pipe', func=userdata_pipe)
    Language.component('test_language_ner_pipe', func=ner_pipe)
    add_vecs_to_vocab(nlp.vocab, sample_vectors)
    nlp.add_pipe('test_language_vector_modification_pipe')
    nlp.add_pipe('test_language_ner_pipe')
    nlp.add_pipe('test_language_userdata_pipe')
    return nlp

@pytest.fixture
def texts():
    if False:
        while True:
            i = 10
    data = ['Hello world.', 'This is spacy.', 'You can use multiprocessing with pipe method.', 'Please try!']
    return data

@pytest.mark.parametrize('n_process', [1, 2])
def test_language_pipe(nlp2, n_process, texts):
    if False:
        for i in range(10):
            print('nop')
    ops = get_current_ops()
    if isinstance(ops, NumpyOps) or n_process < 2:
        texts = texts * 10
        expecteds = [nlp2(text) for text in texts]
        docs = nlp2.pipe(texts, n_process=n_process, batch_size=2)
        for (doc, expected_doc) in zip(docs, expecteds):
            assert_docs_equal(doc, expected_doc)

@pytest.mark.parametrize('n_process', [1, 2])
def test_language_pipe_stream(nlp2, n_process, texts):
    if False:
        while True:
            i = 10
    ops = get_current_ops()
    if isinstance(ops, NumpyOps) or n_process < 2:
        stream_texts = itertools.cycle(texts)
        (texts0, texts1) = itertools.tee(stream_texts)
        expecteds = (nlp2(text) for text in texts0)
        docs = nlp2.pipe(texts1, n_process=n_process, batch_size=2)
        n_fetch = 20
        for (doc, expected_doc) in itertools.islice(zip(docs, expecteds), n_fetch):
            assert_docs_equal(doc, expected_doc)

@pytest.mark.parametrize('n_process', [1, 2])
def test_language_pipe_error_handler(n_process):
    if False:
        print('Hello World!')
    'Test that the error handling of nlp.pipe works well'
    ops = get_current_ops()
    if isinstance(ops, NumpyOps) or n_process < 2:
        nlp = English()
        nlp.add_pipe('merge_subtokens')
        nlp.initialize()
        texts = ['Curious to see what will happen to this text.', 'And this one.']
        with pytest.raises(ValueError):
            nlp(texts[0])
        with pytest.raises(ValueError):
            list(nlp.pipe(texts, n_process=n_process))
        nlp.set_error_handler(raise_error)
        with pytest.raises(ValueError):
            list(nlp.pipe(texts, n_process=n_process))
        nlp.set_error_handler(ignore_error)
        docs = list(nlp.pipe(texts, n_process=n_process))
        assert len(docs) == 0
        nlp(texts[0])

@pytest.mark.parametrize('n_process', [1, 2])
def test_language_pipe_error_handler_custom(en_vocab, n_process):
    if False:
        i = 10
        return i + 15
    'Test the error handling of a custom component that has no pipe method'
    Language.component('my_evil_component', func=evil_component)
    ops = get_current_ops()
    if isinstance(ops, NumpyOps) or n_process < 2:
        nlp = English()
        nlp.add_pipe('my_evil_component')
        texts = ['TEXT 111', 'TEXT 222', 'TEXT 333', 'TEXT 342', 'TEXT 666']
        with pytest.raises(ValueError):
            list(nlp.pipe(texts))
        nlp.set_error_handler(warn_error)
        logger = logging.getLogger('spacy')
        with mock.patch.object(logger, 'warning') as mock_warning:
            docs = list(nlp.pipe(texts, n_process=n_process))
            if n_process == 1:
                mock_warning.assert_called()
                assert mock_warning.call_count == 2
                assert len(docs) + mock_warning.call_count == len(texts)
            assert [doc.text for doc in docs] == ['TEXT 111', 'TEXT 333', 'TEXT 666']

@pytest.mark.parametrize('n_process', [1, 2])
def test_language_pipe_error_handler_input_as_tuples(en_vocab, n_process):
    if False:
        for i in range(10):
            print('nop')
    'Test the error handling of nlp.pipe with input as tuples'
    Language.component('my_evil_component', func=evil_component)
    ops = get_current_ops()
    if isinstance(ops, NumpyOps) or n_process < 2:
        nlp = English()
        nlp.add_pipe('my_evil_component')
        texts = [('TEXT 111', 111), ('TEXT 222', 222), ('TEXT 333', 333), ('TEXT 342', 342), ('TEXT 666', 666)]
        with pytest.raises(ValueError):
            list(nlp.pipe(texts, as_tuples=True))
        nlp.set_error_handler(warn_error)
        logger = logging.getLogger('spacy')
        with mock.patch.object(logger, 'warning') as mock_warning:
            tuples = list(nlp.pipe(texts, as_tuples=True, n_process=n_process))
            if n_process == 1:
                mock_warning.assert_called()
                assert mock_warning.call_count == 2
                assert len(tuples) + mock_warning.call_count == len(texts)
            assert (tuples[0][0].text, tuples[0][1]) == ('TEXT 111', 111)
            assert (tuples[1][0].text, tuples[1][1]) == ('TEXT 333', 333)
            assert (tuples[2][0].text, tuples[2][1]) == ('TEXT 666', 666)

@pytest.mark.parametrize('n_process', [1, 2])
def test_language_pipe_error_handler_pipe(en_vocab, n_process):
    if False:
        while True:
            i = 10
    "Test the error handling of a component's pipe method"
    Language.component('my_perhaps_sentences', func=perhaps_set_sentences)
    Language.component('assert_sents_error', func=assert_sents_error)
    ops = get_current_ops()
    if isinstance(ops, NumpyOps) or n_process < 2:
        texts = [f'{str(i)} is enough. Done' for i in range(100)]
        nlp = English()
        nlp.add_pipe('my_perhaps_sentences')
        nlp.add_pipe('assert_sents_error')
        nlp.initialize()
        with pytest.raises(ValueError):
            docs = list(nlp.pipe(texts, n_process=n_process, batch_size=10))
        nlp.set_error_handler(ignore_error)
        docs = list(nlp.pipe(texts, n_process=n_process, batch_size=10))
        assert len(docs) == 89

@pytest.mark.parametrize('n_process', [1, 2])
def test_language_pipe_error_handler_make_doc_actual(n_process):
    if False:
        print('Hello World!')
    'Test the error handling for make_doc'
    ops = get_current_ops()
    if isinstance(ops, NumpyOps) or n_process < 2:
        nlp = English()
        nlp.max_length = 10
        texts = ['12345678901234567890', '12345'] * 10
        with pytest.raises(ValueError):
            list(nlp.pipe(texts, n_process=n_process))
        nlp.default_error_handler = ignore_error
        if n_process == 1:
            with pytest.raises(ValueError):
                list(nlp.pipe(texts, n_process=n_process))
        else:
            docs = list(nlp.pipe(texts, n_process=n_process))
            assert len(docs) == 0

@pytest.mark.xfail
@pytest.mark.parametrize('n_process', [1, 2])
def test_language_pipe_error_handler_make_doc_preferred(n_process):
    if False:
        i = 10
        return i + 15
    'Test the error handling for make_doc'
    ops = get_current_ops()
    if isinstance(ops, NumpyOps) or n_process < 2:
        nlp = English()
        nlp.max_length = 10
        texts = ['12345678901234567890', '12345'] * 10
        with pytest.raises(ValueError):
            list(nlp.pipe(texts, n_process=n_process))
        nlp.default_error_handler = ignore_error
        docs = list(nlp.pipe(texts, n_process=n_process))
        assert len(docs) == 0

def test_language_from_config_before_after_init():
    if False:
        for i in range(10):
            print('nop')
    name = 'test_language_from_config_before_after_init'
    ran_before = False
    ran_after = False
    ran_after_pipeline = False
    ran_before_init = False
    ran_after_init = False

    @registry.callbacks(f'{name}_before')
    def make_before_creation():
        if False:
            print('Hello World!')

        def before_creation(lang_cls):
            if False:
                print('Hello World!')
            nonlocal ran_before
            ran_before = True
            assert lang_cls is English
            lang_cls.Defaults.foo = 'bar'
            return lang_cls
        return before_creation

    @registry.callbacks(f'{name}_after')
    def make_after_creation():
        if False:
            while True:
                i = 10

        def after_creation(nlp):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal ran_after
            ran_after = True
            assert isinstance(nlp, English)
            assert nlp.pipe_names == []
            assert nlp.Defaults.foo == 'bar'
            nlp.meta['foo'] = 'bar'
            return nlp
        return after_creation

    @registry.callbacks(f'{name}_after_pipeline')
    def make_after_pipeline_creation():
        if False:
            for i in range(10):
                print('nop')

        def after_pipeline_creation(nlp):
            if False:
                return 10
            nonlocal ran_after_pipeline
            ran_after_pipeline = True
            assert isinstance(nlp, English)
            assert nlp.pipe_names == ['sentencizer']
            assert nlp.Defaults.foo == 'bar'
            assert nlp.meta['foo'] == 'bar'
            nlp.meta['bar'] = 'baz'
            return nlp
        return after_pipeline_creation

    @registry.callbacks(f'{name}_before_init')
    def make_before_init():
        if False:
            for i in range(10):
                print('nop')

        def before_init(nlp):
            if False:
                print('Hello World!')
            nonlocal ran_before_init
            ran_before_init = True
            nlp.meta['before_init'] = 'before'
            return nlp
        return before_init

    @registry.callbacks(f'{name}_after_init')
    def make_after_init():
        if False:
            for i in range(10):
                print('nop')

        def after_init(nlp):
            if False:
                i = 10
                return i + 15
            nonlocal ran_after_init
            ran_after_init = True
            nlp.meta['after_init'] = 'after'
            return nlp
        return after_init
    config = {'nlp': {'pipeline': ['sentencizer'], 'before_creation': {'@callbacks': f'{name}_before'}, 'after_creation': {'@callbacks': f'{name}_after'}, 'after_pipeline_creation': {'@callbacks': f'{name}_after_pipeline'}}, 'components': {'sentencizer': {'factory': 'sentencizer'}}, 'initialize': {'before_init': {'@callbacks': f'{name}_before_init'}, 'after_init': {'@callbacks': f'{name}_after_init'}}}
    nlp = English.from_config(config)
    assert nlp.Defaults.foo == 'bar'
    assert nlp.meta['foo'] == 'bar'
    assert nlp.meta['bar'] == 'baz'
    assert 'before_init' not in nlp.meta
    assert 'after_init' not in nlp.meta
    assert nlp.pipe_names == ['sentencizer']
    assert nlp('text')
    nlp.initialize()
    assert nlp.meta['before_init'] == 'before'
    assert nlp.meta['after_init'] == 'after'
    assert all([ran_before, ran_after, ran_after_pipeline, ran_before_init, ran_after_init])

def test_language_from_config_before_after_init_invalid():
    if False:
        i = 10
        return i + 15
    "Check that an error is raised if function doesn't return nlp."
    name = 'test_language_from_config_before_after_init_invalid'
    registry.callbacks(f'{name}_before1', func=lambda : lambda nlp: None)
    registry.callbacks(f'{name}_before2', func=lambda : lambda nlp: nlp())
    registry.callbacks(f'{name}_after1', func=lambda : lambda nlp: None)
    registry.callbacks(f'{name}_after1', func=lambda : lambda nlp: English)
    for callback_name in [f'{name}_before1', f'{name}_before2']:
        config = {'nlp': {'before_creation': {'@callbacks': callback_name}}}
        with pytest.raises(ValueError):
            English.from_config(config)
    for callback_name in [f'{name}_after1', f'{name}_after2']:
        config = {'nlp': {'after_creation': {'@callbacks': callback_name}}}
        with pytest.raises(ValueError):
            English.from_config(config)
    for callback_name in [f'{name}_after1', f'{name}_after2']:
        config = {'nlp': {'after_pipeline_creation': {'@callbacks': callback_name}}}
        with pytest.raises(ValueError):
            English.from_config(config)

def test_language_whitespace_tokenizer():
    if False:
        return 10
    'Test the custom whitespace tokenizer from the docs.'

    class WhitespaceTokenizer:

        def __init__(self, vocab):
            if False:
                return 10
            self.vocab = vocab

        def __call__(self, text):
            if False:
                print('Hello World!')
            words = text.split(' ')
            spaces = [True] * len(words)
            for (i, word) in enumerate(words):
                if word == '':
                    words[i] = ' '
                    spaces[i] = False
            if words[-1] == ' ':
                words = words[0:-1]
                spaces = spaces[0:-1]
            else:
                spaces[-1] = False
            return Doc(self.vocab, words=words, spaces=spaces)
    nlp = spacy.blank('en')
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    text = "   What's happened to    me? he thought. It wasn't a dream.    "
    doc = nlp(text)
    assert doc.text == text

def test_language_custom_tokenizer():
    if False:
        print('Hello World!')
    'Test that a fully custom tokenizer can be plugged in via the registry.'
    name = 'test_language_custom_tokenizer'

    class CustomTokenizer:
        """Dummy "tokenizer" that splits on spaces and adds prefix to each word."""

        def __init__(self, nlp, prefix):
            if False:
                return 10
            self.vocab = nlp.vocab
            self.prefix = prefix

        def __call__(self, text):
            if False:
                return 10
            words = [f'{self.prefix}{word}' for word in text.split(' ')]
            return Doc(self.vocab, words=words)

    @registry.tokenizers(name)
    def custom_create_tokenizer(prefix: str='_'):
        if False:
            for i in range(10):
                print('nop')

        def create_tokenizer(nlp):
            if False:
                for i in range(10):
                    print('nop')
            return CustomTokenizer(nlp, prefix=prefix)
        return create_tokenizer
    config = {'nlp': {'tokenizer': {'@tokenizers': name}}}
    nlp = English.from_config(config)
    doc = nlp('hello world')
    assert [t.text for t in doc] == ['_hello', '_world']
    doc = list(nlp.pipe(['hello world']))[0]
    assert [t.text for t in doc] == ['_hello', '_world']

def test_language_from_config_invalid_lang():
    if False:
        print('Hello World!')
    'Test that calling Language.from_config raises an error and lang defined\n    in config needs to match language-specific subclasses.'
    config = {'nlp': {'lang': 'en'}}
    with pytest.raises(ValueError):
        Language.from_config(config)
    with pytest.raises(ValueError):
        German.from_config(config)

def test_spacy_blank():
    if False:
        for i in range(10):
            print('nop')
    nlp = spacy.blank('en')
    assert nlp.config['training']['dropout'] == 0.1
    config = {'training': {'dropout': 0.2}}
    meta = {'name': 'my_custom_model'}
    nlp = spacy.blank('en', config=config, meta=meta)
    assert nlp.config['training']['dropout'] == 0.2
    assert nlp.meta['name'] == 'my_custom_model'

@pytest.mark.parametrize('lang,target', [('en', 'en'), ('fra', 'fr'), ('fre', 'fr'), ('iw', 'he'), ('mo', 'ro'), ('mul', 'xx'), ('no', 'nb'), ('pt-BR', 'pt'), ('xx', 'xx'), ('zh-Hans', 'zh'), ('zh-Hant', None), ('zxx', None)])
def test_language_matching(lang, target):
    if False:
        i = 10
        return i + 15
    '\n    Test that we can look up languages by equivalent or nearly-equivalent\n    language codes.\n    '
    assert find_matching_language(lang) == target

@pytest.mark.parametrize('lang,target', [('en', 'en'), ('fra', 'fr'), ('fre', 'fr'), ('iw', 'he'), ('mo', 'ro'), ('mul', 'xx'), ('no', 'nb'), ('pt-BR', 'pt'), ('xx', 'xx'), ('zh-Hans', 'zh')])
def test_blank_languages(lang, target):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that we can get spacy.blank in various languages, including codes\n    that are defined to be equivalent or that match by CLDR language matching.\n    '
    nlp = spacy.blank(lang)
    assert nlp.lang == target

@pytest.mark.parametrize('value', [False, None, ['x', 'y'], Language, Vocab])
def test_language_init_invalid_vocab(value):
    if False:
        while True:
            i = 10
    err_fragment = 'invalid value'
    with pytest.raises(ValueError) as e:
        Language(value)
    assert err_fragment in str(e.value)

def test_language_source_and_vectors(nlp2):
    if False:
        for i in range(10):
            print('nop')
    nlp = Language(Vocab())
    textcat = nlp.add_pipe('textcat')
    for label in ('POSITIVE', 'NEGATIVE'):
        textcat.add_label(label)
    nlp.initialize()
    long_string = 'thisisalongstring'
    assert long_string not in nlp.vocab.strings
    assert long_string not in nlp2.vocab.strings
    nlp.vocab.strings.add(long_string)
    assert nlp.vocab.vectors.to_bytes() != nlp2.vocab.vectors.to_bytes()
    vectors_bytes = nlp.vocab.vectors.to_bytes()
    with pytest.warns(UserWarning):
        nlp2.add_pipe('textcat', name='textcat2', source=nlp)
    assert long_string in nlp2.vocab.strings
    assert nlp.vocab.vectors.to_bytes() == vectors_bytes

@pytest.mark.parametrize('n_process', [1, 2])
def test_pass_doc_to_pipeline(nlp, n_process):
    if False:
        while True:
            i = 10
    texts = ['cats', 'dogs', 'guinea pigs']
    docs = [nlp.make_doc(text) for text in texts]
    assert not any((len(doc.cats) for doc in docs))
    doc = nlp(docs[0])
    assert doc.text == texts[0]
    assert len(doc.cats) > 0
    if isinstance(get_current_ops(), NumpyOps) or n_process < 2:
        docs = nlp.pipe(docs, n_process=n_process)
        assert [doc.text for doc in docs] == texts
        assert all((len(doc.cats) for doc in docs))

def test_invalid_arg_to_pipeline(nlp):
    if False:
        while True:
            i = 10
    str_list = ['This is a text.', 'This is another.']
    with pytest.raises(ValueError):
        nlp(str_list)
    assert len(list(nlp.pipe(str_list))) == 2
    int_list = [1, 2, 3]
    with pytest.raises(ValueError):
        list(nlp.pipe(int_list))
    with pytest.raises(ValueError):
        nlp(int_list)

@pytest.mark.skipif(not isinstance(get_current_ops(), CupyOps), reason='test requires GPU')
def test_multiprocessing_gpu_warning(nlp2, texts):
    if False:
        while True:
            i = 10
    texts = texts * 10
    docs = nlp2.pipe(texts, n_process=2, batch_size=2)
    with pytest.warns(UserWarning, match='multiprocessing with GPU models'):
        with pytest.raises(ValueError):
            for _ in docs:
                pass

def test_dot_in_factory_names(nlp):
    if False:
        while True:
            i = 10
    Language.component('my_evil_component', func=evil_component)
    nlp.add_pipe('my_evil_component')
    with pytest.raises(ValueError, match='not permitted'):
        Language.component('my.evil.component.v1', func=evil_component)
    with pytest.raises(ValueError, match='not permitted'):
        Language.factory('my.evil.component.v1', func=evil_component)

def test_component_return():
    if False:
        while True:
            i = 10
    'Test that an error is raised if components return a type other than a\n    doc.'
    nlp = English()

    @Language.component('test_component_good_pipe')
    def good_pipe(doc):
        if False:
            while True:
                i = 10
        return doc
    nlp.add_pipe('test_component_good_pipe')
    nlp('text')
    nlp.remove_pipe('test_component_good_pipe')

    @Language.component('test_component_bad_pipe')
    def bad_pipe(doc):
        if False:
            for i in range(10):
                print('nop')
        return doc.text
    nlp.add_pipe('test_component_bad_pipe')
    with pytest.raises(ValueError, match='instead of a Doc'):
        nlp('text')