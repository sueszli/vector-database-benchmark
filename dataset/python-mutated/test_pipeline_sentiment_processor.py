import pytest
import stanza
from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from stanza.tests import *
pytestmark = [pytest.mark.pipeline, pytest.mark.travis]
EN_DOCS = ['Ragavan is terrible and should go away.', 'Today is okay.', "Urza's Saga is great."]
EN_DOC = '  '.join(EN_DOCS)
EXPECTED = [0, 1, 2]

class TestSentimentPipeline:

    @pytest.fixture(scope='class')
    def pipeline(self):
        if False:
            print('Hello World!')
        '\n        A reusable pipeline with the NER module\n        '
        return stanza.Pipeline(dir=TEST_MODELS_DIR, processors='tokenize,sentiment')

    def test_simple(self, pipeline):
        if False:
            for i in range(10):
                print('nop')
        results = []
        for text in EN_DOCS:
            doc = pipeline(text)
            assert len(doc.sentences) == 1
            results.append(doc.sentences[0].sentiment)
        assert EXPECTED == results

    def test_multiple_sentences(self, pipeline):
        if False:
            return 10
        doc = pipeline(EN_DOC)
        assert len(doc.sentences) == 3
        results = [sentence.sentiment for sentence in doc.sentences]
        assert EXPECTED == results

    def test_empty_text(self, pipeline):
        if False:
            print('Hello World!')
        '\n        Test empty text and a text which might get reduced to empty text by removing dashes\n        '
        doc = pipeline('')
        assert len(doc.sentences) == 0
        doc = pipeline('--')
        assert len(doc.sentences) == 1