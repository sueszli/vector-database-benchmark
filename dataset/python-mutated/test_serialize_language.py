import pickle
import re
import pytest
from spacy.lang.en import English
from spacy.lang.it import Italian
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy.training import Example
from spacy.util import load_config_from_str
from ..util import make_tempdir

@pytest.fixture
def meta_data():
    if False:
        print('Hello World!')
    return {'name': 'name-in-fixture', 'version': 'version-in-fixture', 'description': 'description-in-fixture', 'author': 'author-in-fixture', 'email': 'email-in-fixture', 'url': 'url-in-fixture', 'license': 'license-in-fixture', 'vectors': {'width': 0, 'vectors': 0, 'keys': 0, 'name': None}}

@pytest.mark.issue(2482)
def test_issue2482():
    if False:
        for i in range(10):
            print('nop')
    'Test we can serialize and deserialize a blank NER or parser model.'
    nlp = Italian()
    nlp.add_pipe('ner')
    b = nlp.to_bytes()
    Italian().from_bytes(b)
CONFIG_ISSUE_6950 = '\n[nlp]\nlang = "en"\npipeline = ["tok2vec", "tagger"]\n\n[components]\n\n[components.tok2vec]\nfactory = "tok2vec"\n\n[components.tok2vec.model]\n@architectures = "spacy.Tok2Vec.v1"\n\n[components.tok2vec.model.embed]\n@architectures = "spacy.MultiHashEmbed.v1"\nwidth = ${components.tok2vec.model.encode:width}\nattrs = ["NORM","PREFIX","SUFFIX","SHAPE"]\nrows = [5000,2500,2500,2500]\ninclude_static_vectors = false\n\n[components.tok2vec.model.encode]\n@architectures = "spacy.MaxoutWindowEncoder.v1"\nwidth = 96\ndepth = 4\nwindow_size = 1\nmaxout_pieces = 3\n\n[components.ner]\nfactory = "ner"\n\n[components.tagger]\nfactory = "tagger"\n\n[components.tagger.model]\n@architectures = "spacy.Tagger.v2"\nnO = null\n\n[components.tagger.model.tok2vec]\n@architectures = "spacy.Tok2VecListener.v1"\nwidth = ${components.tok2vec.model.encode:width}\nupstream = "*"\n'

@pytest.mark.issue(6950)
def test_issue6950():
    if False:
        for i in range(10):
            print('nop')
    "Test that the nlp object with initialized tok2vec with listeners pickles\n    correctly (and doesn't have lambdas).\n    "
    nlp = English.from_config(load_config_from_str(CONFIG_ISSUE_6950))
    nlp.initialize(lambda : [Example.from_dict(nlp.make_doc('hello'), {'tags': ['V']})])
    pickle.dumps(nlp)
    nlp('hello')
    pickle.dumps(nlp)

def test_serialize_language_meta_disk(meta_data):
    if False:
        i = 10
        return i + 15
    language = Language(meta=meta_data)
    with make_tempdir() as d:
        language.to_disk(d)
        new_language = Language().from_disk(d)
    assert new_language.meta == language.meta

def test_serialize_with_custom_tokenizer():
    if False:
        for i in range(10):
            print('nop')
    'Test that serialization with custom tokenizer works without token_match.\n    See: https://support.prodi.gy/t/how-to-save-a-custom-tokenizer/661/2\n    '
    prefix_re = re.compile('1/|2/|:[0-9][0-9][A-K]:|:[0-9][0-9]:')
    suffix_re = re.compile('')
    infix_re = re.compile('[~]')

    def custom_tokenizer(nlp):
        if False:
            while True:
                i = 10
        return Tokenizer(nlp.vocab, {}, prefix_search=prefix_re.search, suffix_search=suffix_re.search, infix_finditer=infix_re.finditer)
    nlp = Language()
    nlp.tokenizer = custom_tokenizer(nlp)
    with make_tempdir() as d:
        nlp.to_disk(d)

def test_serialize_language_exclude(meta_data):
    if False:
        for i in range(10):
            print('nop')
    name = 'name-in-fixture'
    nlp = Language(meta=meta_data)
    assert nlp.meta['name'] == name
    new_nlp = Language().from_bytes(nlp.to_bytes())
    assert new_nlp.meta['name'] == name
    new_nlp = Language().from_bytes(nlp.to_bytes(), exclude=['meta'])
    assert not new_nlp.meta['name'] == name
    new_nlp = Language().from_bytes(nlp.to_bytes(exclude=['meta']))
    assert not new_nlp.meta['name'] == name