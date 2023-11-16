from typing import Callable, Iterable, Iterator
import pytest
from thinc.api import Config
from spacy.lang.en import English
from spacy.language import Language
from spacy.training import Example
from spacy.training.loop import train
from spacy.util import load_model_from_config, registry

@pytest.fixture
def config_str():
    if False:
        print('Hello World!')
    return '\n    [nlp]\n    lang = "en"\n    pipeline = ["sentencizer","assert_sents"]\n    disabled = []\n    before_creation = null\n    after_creation = null\n    after_pipeline_creation = null\n    batch_size = 1000\n    tokenizer = {"@tokenizers":"spacy.Tokenizer.v1"}\n\n    [components]\n\n    [components.assert_sents]\n    factory = "assert_sents"\n\n    [components.sentencizer]\n    factory = "sentencizer"\n    punct_chars = null\n\n    [training]\n    dev_corpus = "corpora.dev"\n    train_corpus = "corpora.train"\n    annotating_components = ["sentencizer"]\n    max_steps = 2\n\n    [corpora]\n\n    [corpora.dev]\n    @readers = "unannotated_corpus"\n\n    [corpora.train]\n    @readers = "unannotated_corpus"\n    '

def test_annotates_on_update():
    if False:
        print('Hello World!')

    @Language.factory('assert_sents', default_config={})
    def assert_sents(nlp, name):
        if False:
            for i in range(10):
                print('nop')
        return AssertSents(name)

    class AssertSents:

        def __init__(self, name, **cfg):
            if False:
                print('Hello World!')
            self.name = name
            pass

        def __call__(self, doc):
            if False:
                print('Hello World!')
            if not doc.has_annotation('SENT_START'):
                raise ValueError('No sents')
            return doc

        def update(self, examples, *, drop=0.0, sgd=None, losses=None):
            if False:
                return 10
            for example in examples:
                if not example.predicted.has_annotation('SENT_START'):
                    raise ValueError('No sents')
            return {}
    nlp = English()
    nlp.add_pipe('sentencizer')
    nlp.add_pipe('assert_sents')
    nlp('This is a sentence.')
    examples = []
    for text in ['a a', 'b b', 'c c']:
        examples.append(Example(nlp.make_doc(text), nlp(text)))
    for example in examples:
        assert not example.predicted.has_annotation('SENT_START')
    with pytest.raises(ValueError):
        nlp.update(examples)
    nlp.update(examples, annotates=['sentencizer'])

def test_annotating_components_from_config(config_str):
    if False:
        print('Hello World!')

    @registry.readers('unannotated_corpus')
    def create_unannotated_corpus() -> Callable[[Language], Iterable[Example]]:
        if False:
            i = 10
            return i + 15
        return UnannotatedCorpus()

    class UnannotatedCorpus:

        def __call__(self, nlp: Language) -> Iterator[Example]:
            if False:
                for i in range(10):
                    print('nop')
            for text in ['a a', 'b b', 'c c']:
                doc = nlp.make_doc(text)
                yield Example(doc, doc)
    orig_config = Config().from_str(config_str)
    nlp = load_model_from_config(orig_config, auto_fill=True, validate=True)
    assert nlp.config['training']['annotating_components'] == ['sentencizer']
    train(nlp)
    nlp.config['training']['annotating_components'] = []
    with pytest.raises(ValueError):
        train(nlp)