import pytest
from haystack.preview import Document
from haystack.preview.components.preprocessors import DocumentSplitter

class TestDocumentSplitter:

    @pytest.mark.unit
    def test_non_text_document(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError, match='DocumentSplitter only works with text documents but document.content for document ID'):
            splitter = DocumentSplitter()
            splitter.run(documents=[Document()])

    @pytest.mark.unit
    def test_single_doc(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError, match='DocumentSplitter expects a List of Documents as input.'):
            splitter = DocumentSplitter()
            splitter.run(documents=Document())

    @pytest.mark.unit
    def test_empty_list(self):
        if False:
            while True:
                i = 10
        splitter = DocumentSplitter()
        res = splitter.run(documents=[])
        assert res == {'documents': []}

    @pytest.mark.unit
    def test_unsupported_split_by(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError, match="split_by must be one of 'word', 'sentence' or 'passage'."):
            DocumentSplitter(split_by='unsupported')

    @pytest.mark.unit
    def test_unsupported_split_length(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError, match='split_length must be greater than 0.'):
            DocumentSplitter(split_length=0)

    @pytest.mark.unit
    def test_unsupported_split_overlap(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError, match='split_overlap must be greater than or equal to 0.'):
            DocumentSplitter(split_overlap=-1)

    @pytest.mark.unit
    def test_split_by_word(self):
        if False:
            print('Hello World!')
        splitter = DocumentSplitter(split_by='word', split_length=10)
        result = splitter.run(documents=[Document(content='This is a text with some words. There is a second sentence. And there is a third sentence.')])
        assert len(result['documents']) == 2
        assert result['documents'][0].content == 'This is a text with some words. There is a '
        assert result['documents'][1].content == 'second sentence. And there is a third sentence.'

    @pytest.mark.unit
    def test_split_by_word_multiple_input_docs(self):
        if False:
            return 10
        splitter = DocumentSplitter(split_by='word', split_length=10)
        result = splitter.run(documents=[Document(content='This is a text with some words. There is a second sentence. And there is a third sentence.'), Document(content='This is a different text with some words. There is a second sentence. And there is a third sentence. And there is a fourth sentence.')])
        assert len(result['documents']) == 5
        assert result['documents'][0].content == 'This is a text with some words. There is a '
        assert result['documents'][1].content == 'second sentence. And there is a third sentence.'
        assert result['documents'][2].content == 'This is a different text with some words. There is '
        assert result['documents'][3].content == 'a second sentence. And there is a third sentence. And '
        assert result['documents'][4].content == 'there is a fourth sentence.'

    @pytest.mark.unit
    def test_split_by_sentence(self):
        if False:
            return 10
        splitter = DocumentSplitter(split_by='sentence', split_length=1)
        result = splitter.run(documents=[Document(content='This is a text with some words. There is a second sentence. And there is a third sentence.')])
        assert len(result['documents']) == 3
        assert result['documents'][0].content == 'This is a text with some words.'
        assert result['documents'][1].content == ' There is a second sentence.'
        assert result['documents'][2].content == ' And there is a third sentence.'

    @pytest.mark.unit
    def test_split_by_passage(self):
        if False:
            i = 10
            return i + 15
        splitter = DocumentSplitter(split_by='passage', split_length=1)
        result = splitter.run(documents=[Document(content='This is a text with some words. There is a second sentence.\n\nAnd there is a third sentence.\n\n And another passage.')])
        assert len(result['documents']) == 3
        assert result['documents'][0].content == 'This is a text with some words. There is a second sentence.\n\n'
        assert result['documents'][1].content == 'And there is a third sentence.\n\n'
        assert result['documents'][2].content == ' And another passage.'

    @pytest.mark.unit
    def test_split_by_word_with_overlap(self):
        if False:
            return 10
        splitter = DocumentSplitter(split_by='word', split_length=10, split_overlap=2)
        result = splitter.run(documents=[Document(content='This is a text with some words. There is a second sentence. And there is a third sentence.')])
        assert len(result['documents']) == 2
        assert result['documents'][0].content == 'This is a text with some words. There is a '
        assert result['documents'][1].content == 'is a second sentence. And there is a third sentence.'

    @pytest.mark.unit
    def test_source_id_stored_in_metadata(self):
        if False:
            print('Hello World!')
        splitter = DocumentSplitter(split_by='word', split_length=10)
        doc1 = Document(content='This is a text with some words.')
        doc2 = Document(content='This is a different text with some words.')
        result = splitter.run(documents=[doc1, doc2])
        assert result['documents'][0].meta['source_id'] == doc1.id
        assert result['documents'][1].meta['source_id'] == doc2.id

    @pytest.mark.unit
    def test_copy_metadata(self):
        if False:
            while True:
                i = 10
        splitter = DocumentSplitter(split_by='word', split_length=10)
        documents = [Document(content='Text.', meta={'name': 'doc 0'}), Document(content='Text.', meta={'name': 'doc 1'})]
        result = splitter.run(documents=documents)
        assert len(result['documents']) == 2
        assert result['documents'][0].id != result['documents'][1].id
        for (doc, split_doc) in zip(documents, result['documents']):
            assert doc.meta.items() <= split_doc.meta.items()
            assert split_doc.content == 'Text.'