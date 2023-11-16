import logging
import pytest
from haystack.preview import Document
from haystack.preview.components.preprocessors import DocumentCleaner

class TestDocumentCleaner:

    @pytest.mark.unit
    def test_init(self):
        if False:
            print('Hello World!')
        cleaner = DocumentCleaner()
        assert cleaner.remove_empty_lines is True
        assert cleaner.remove_extra_whitespaces is True
        assert cleaner.remove_repeated_substrings is False
        assert cleaner.remove_substrings is None
        assert cleaner.remove_regex is None

    @pytest.mark.unit
    def test_non_text_document(self, caplog):
        if False:
            while True:
                i = 10
        with caplog.at_level(logging.WARNING):
            cleaner = DocumentCleaner()
            cleaner.run(documents=[Document()])
            assert 'DocumentCleaner only cleans text documents but document.content for document ID' in caplog.text

    @pytest.mark.unit
    def test_single_document(self):
        if False:
            return 10
        with pytest.raises(TypeError, match='DocumentCleaner expects a List of Documents as input.'):
            cleaner = DocumentCleaner()
            cleaner.run(documents=Document())

    @pytest.mark.unit
    def test_empty_list(self):
        if False:
            while True:
                i = 10
        cleaner = DocumentCleaner()
        result = cleaner.run(documents=[])
        assert result == {'documents': []}

    @pytest.mark.unit
    def test_remove_empty_lines(self):
        if False:
            print('Hello World!')
        cleaner = DocumentCleaner(remove_extra_whitespaces=False)
        result = cleaner.run(documents=[Document(content='This is a text with some words. There is a second sentence. And there is a third sentence.')])
        assert len(result['documents']) == 1
        assert result['documents'][0].content == 'This is a text with some words. There is a second sentence. And there is a third sentence.'

    @pytest.mark.unit
    def test_remove_whitespaces(self):
        if False:
            i = 10
            return i + 15
        cleaner = DocumentCleaner(remove_empty_lines=False)
        result = cleaner.run(documents=[Document(content=' This is a text with some words. There is a second sentence.  And there  is a third sentence. ')])
        assert len(result['documents']) == 1
        assert result['documents'][0].content == 'This is a text with some words. There is a second sentence. And there is a third sentence.'

    @pytest.mark.unit
    def test_remove_substrings(self):
        if False:
            print('Hello World!')
        cleaner = DocumentCleaner(remove_substrings=['This', 'A', 'words', '🪲'])
        result = cleaner.run(documents=[Document(content='This is a text with some words.🪲')])
        assert len(result['documents']) == 1
        assert result['documents'][0].content == ' is a text with some .'

    @pytest.mark.unit
    def test_remove_regex(self):
        if False:
            i = 10
            return i + 15
        cleaner = DocumentCleaner(remove_regex='\\s\\s+')
        result = cleaner.run(documents=[Document(content='This is a  text with   some words.')])
        assert len(result['documents']) == 1
        assert result['documents'][0].content == 'This is a text with some words.'

    @pytest.mark.unit
    def test_remove_repeated_substrings(self):
        if False:
            while True:
                i = 10
        cleaner = DocumentCleaner(remove_empty_lines=False, remove_extra_whitespaces=False, remove_repeated_substrings=True)
        text = 'First Page\x0cThis is a header.\n        Page  of\n        2\n        4\n        Lorem ipsum dolor sit amet\n        This is a footer number 1\n        This is footer number 2\x0cThis is a header.\n        Page  of\n        3\n        4\n        Sid ut perspiciatis unde\n        This is a footer number 1\n        This is footer number 2\x0cThis is a header.\n        Page  of\n        4\n        4\n        Sed do eiusmod tempor.\n        This is a footer number 1\n        This is footer number 2'
        expected_text = 'First Page\x0c 2\n        4\n        Lorem ipsum dolor sit amet\x0c 3\n        4\n        Sid ut perspiciatis unde\x0c 4\n        4\n        Sed do eiusmod tempor.'
        result = cleaner.run(documents=[Document(content=text)])
        assert result['documents'][0].content == expected_text

    @pytest.mark.unit
    def test_copy_metadata(self):
        if False:
            for i in range(10):
                print('nop')
        cleaner = DocumentCleaner()
        documents = [Document(content='Text. ', meta={'name': 'doc 0'}), Document(content='Text. ', meta={'name': 'doc 1'})]
        result = cleaner.run(documents=documents)
        assert len(result['documents']) == 2
        assert result['documents'][0].id != result['documents'][1].id
        for (doc, cleaned_doc) in zip(documents, result['documents']):
            assert doc.meta == cleaned_doc.meta
            assert cleaned_doc.content == 'Text.'