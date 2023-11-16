import pytest
import haystack
from haystack.schema import Document
from haystack.nodes import TransformersTranslator
ORIGINAL_TEXT = 'TEST QUERY'
TRANSLATION = 'MOCK TRANSLATION'

class MockTokenizer:

    @classmethod
    def from_pretrained(cls, *a, **k):
        if False:
            print('Hello World!')
        return cls()

    def __call__(self, *a, **k):
        if False:
            return 10
        return self

    def to(self, *a, **k):
        if False:
            i = 10
            return i + 15
        return {}

    def batch_decode(self, *a, **k):
        if False:
            print('Hello World!')
        return [TRANSLATION]

class MockModel:

    @classmethod
    def from_pretrained(cls, *a, **k):
        if False:
            for i in range(10):
                print('nop')
        return cls()

    def generate(self, *a, **k):
        if False:
            while True:
                i = 10
        return None

    def to(self, *a, **k):
        if False:
            while True:
                i = 10
        return None

@pytest.fixture
def mock_models(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr(haystack.nodes.translator.transformers, 'AutoModelForSeq2SeqLM', MockModel)
    monkeypatch.setattr(haystack.nodes.translator.transformers, 'AutoTokenizer', MockTokenizer)

@pytest.fixture
def en_to_de_translator(mock_models) -> TransformersTranslator:
    if False:
        i = 10
        return i + 15
    return TransformersTranslator(model_name_or_path='irrelevant/anyway')

@pytest.fixture
def de_to_en_translator(mock_models) -> TransformersTranslator:
    if False:
        while True:
            i = 10
    return TransformersTranslator(model_name_or_path='irrelevant/anyway')

@pytest.mark.unit
def test_translator_with_query(en_to_de_translator):
    if False:
        print('Hello World!')
    assert en_to_de_translator.translate(query=ORIGINAL_TEXT) == TRANSLATION

@pytest.mark.unit
def test_translator_with_list(en_to_de_translator):
    if False:
        while True:
            i = 10
    assert en_to_de_translator.translate(documents=[ORIGINAL_TEXT])[0] == TRANSLATION

@pytest.mark.unit
def test_translator_with_document(en_to_de_translator):
    if False:
        for i in range(10):
            print('nop')
    assert en_to_de_translator.translate(documents=[Document(content=ORIGINAL_TEXT)])[0].content == TRANSLATION

@pytest.mark.unit
def test_translator_with_document_preserves_original(en_to_de_translator):
    if False:
        print('Hello World!')
    original_document = Document(content=ORIGINAL_TEXT)
    en_to_de_translator.translate(documents=[original_document])
    assert original_document.content == ORIGINAL_TEXT

@pytest.mark.unit
def test_translator_with_dictionary(en_to_de_translator):
    if False:
        print('Hello World!')
    assert en_to_de_translator.translate(documents=[{'content': ORIGINAL_TEXT}])[0]['content'] == TRANSLATION

@pytest.mark.unit
def test_translator_with_dictionary_preserves_original(en_to_de_translator):
    if False:
        i = 10
        return i + 15
    original_document = {'content': ORIGINAL_TEXT}
    en_to_de_translator.translate(documents=[original_document])
    assert original_document['content'] == ORIGINAL_TEXT

@pytest.mark.unit
def test_translator_with_dictionary_with_dict_key(en_to_de_translator):
    if False:
        i = 10
        return i + 15
    assert en_to_de_translator.translate(documents=[{'key': ORIGINAL_TEXT}], dict_key='key')[0]['key'] == TRANSLATION

@pytest.mark.unit
def test_translator_with_empty_original(en_to_de_translator):
    if False:
        i = 10
        return i + 15
    with pytest.raises(AttributeError):
        en_to_de_translator.translate()

@pytest.mark.unit
def test_translator_with_query_and_documents(en_to_de_translator):
    if False:
        return 10
    with pytest.raises(AttributeError):
        en_to_de_translator.translate(query=ORIGINAL_TEXT, documents=[ORIGINAL_TEXT])

@pytest.mark.unit
def test_translator_with_dict_without_text_key(en_to_de_translator):
    if False:
        print('Hello World!')
    with pytest.raises(AttributeError):
        en_to_de_translator.translate(documents=[{'text1': ORIGINAL_TEXT}])

@pytest.mark.unit
def test_translator_with_dict_with_non_string_value(en_to_de_translator):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(AttributeError):
        en_to_de_translator.translate(documents=[{'text': 123}])