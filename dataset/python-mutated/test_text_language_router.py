import logging
import pytest
from haystack.preview import Document
from haystack.preview.components.routers import TextLanguageRouter

class TestTextLanguageRouter:

    @pytest.mark.unit
    def test_non_string_input(self):
        if False:
            return 10
        with pytest.raises(TypeError, match='TextLanguageRouter expects a str as input.'):
            classifier = TextLanguageRouter()
            classifier.run(text=Document(content='This is an english sentence.'))

    @pytest.mark.unit
    def test_list_of_string(self):
        if False:
            return 10
        with pytest.raises(TypeError, match='TextLanguageRouter expects a str as input.'):
            classifier = TextLanguageRouter()
            classifier.run(text=['This is an english sentence.'])

    @pytest.mark.unit
    def test_empty_string(self):
        if False:
            i = 10
            return i + 15
        classifier = TextLanguageRouter()
        result = classifier.run(text='')
        assert result == {'unmatched': ''}

    @pytest.mark.unit
    def test_detect_language(self):
        if False:
            for i in range(10):
                print('nop')
        classifier = TextLanguageRouter()
        detected_language = classifier.detect_language('This is an english sentence.')
        assert detected_language == 'en'

    @pytest.mark.unit
    def test_route_to_en(self):
        if False:
            i = 10
            return i + 15
        classifier = TextLanguageRouter()
        english_sentence = 'This is an english sentence.'
        result = classifier.run(text=english_sentence)
        assert result == {'en': english_sentence}

    @pytest.mark.unit
    def test_route_to_unmatched(self):
        if False:
            while True:
                i = 10
        classifier = TextLanguageRouter()
        german_sentence = 'Ein deutscher Satz ohne Verb.'
        result = classifier.run(text=german_sentence)
        assert result == {'unmatched': german_sentence}

    @pytest.mark.unit
    def test_warning_if_no_language_detected(self, caplog):
        if False:
            for i in range(10):
                print('nop')
        with caplog.at_level(logging.WARNING):
            classifier = TextLanguageRouter()
            classifier.run(text='.')
            assert 'Langdetect cannot detect the language of text: .' in caplog.text