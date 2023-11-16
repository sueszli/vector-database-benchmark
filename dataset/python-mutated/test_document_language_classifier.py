import logging
import pytest
from haystack.preview import Document
from haystack.preview.components.classifiers import DocumentLanguageClassifier

class TestDocumentLanguageClassifier:

    @pytest.mark.unit
    def test_init(self):
        if False:
            return 10
        component = DocumentLanguageClassifier()
        assert component.languages == ['en']

    @pytest.mark.unit
    def test_non_document_input(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(TypeError, match='DocumentLanguageClassifier expects a list of Document as input.'):
            classifier = DocumentLanguageClassifier()
            classifier.run(documents='This is an english sentence.')

    @pytest.mark.unit
    def test_single_document(self):
        if False:
            while True:
                i = 10
        with pytest.raises(TypeError, match='DocumentLanguageClassifier expects a list of Document as input.'):
            classifier = DocumentLanguageClassifier()
            classifier.run(documents=Document(content='This is an english sentence.'))

    @pytest.mark.unit
    def test_empty_list(self):
        if False:
            print('Hello World!')
        classifier = DocumentLanguageClassifier()
        result = classifier.run(documents=[])
        assert result == {'documents': []}

    @pytest.mark.unit
    def test_detect_language(self):
        if False:
            for i in range(10):
                print('nop')
        classifier = DocumentLanguageClassifier()
        detected_language = classifier.detect_language(Document(content='This is an english sentence.'))
        assert detected_language == 'en'

    @pytest.mark.unit
    def test_classify_as_en_and_unmatched(self):
        if False:
            i = 10
            return i + 15
        classifier = DocumentLanguageClassifier()
        english_document = Document(content='This is an english sentence.')
        german_document = Document(content='Ein deutscher Satz ohne Verb.')
        result = classifier.run(documents=[english_document, german_document])
        assert result['documents'][0].meta['language'] == 'en'
        assert result['documents'][1].meta['language'] == 'unmatched'

    @pytest.mark.unit
    def test_warning_if_no_language_detected(self, caplog):
        if False:
            return 10
        with caplog.at_level(logging.WARNING):
            classifier = DocumentLanguageClassifier()
            classifier.run(documents=[Document(content='.')])
            assert 'Langdetect cannot detect the language of Document with id' in caplog.text