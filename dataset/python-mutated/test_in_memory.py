import logging
from unittest.mock import patch
import pandas as pd
import pytest
from haystack.preview import Document
from haystack.preview.document_stores import InMemoryDocumentStore, DocumentStoreError
from haystack.preview.testing.document_store import DocumentStoreBaseTests

class TestMemoryDocumentStore(DocumentStoreBaseTests):
    """
    Test InMemoryDocumentStore's specific features
    """

    @pytest.fixture
    def docstore(self) -> InMemoryDocumentStore:
        if False:
            return 10
        return InMemoryDocumentStore()

    @pytest.mark.unit
    def test_to_dict(self):
        if False:
            return 10
        store = InMemoryDocumentStore()
        data = store.to_dict()
        assert data == {'type': 'InMemoryDocumentStore', 'init_parameters': {'bm25_tokenization_regex': '(?u)\\b\\w\\w+\\b', 'bm25_algorithm': 'BM25Okapi', 'bm25_parameters': {}, 'embedding_similarity_function': 'dot_product'}}

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        if False:
            while True:
                i = 10
        store = InMemoryDocumentStore(bm25_tokenization_regex='custom_regex', bm25_algorithm='BM25Plus', bm25_parameters={'key': 'value'}, embedding_similarity_function='cosine')
        data = store.to_dict()
        assert data == {'type': 'InMemoryDocumentStore', 'init_parameters': {'bm25_tokenization_regex': 'custom_regex', 'bm25_algorithm': 'BM25Plus', 'bm25_parameters': {'key': 'value'}, 'embedding_similarity_function': 'cosine'}}

    @pytest.mark.unit
    @patch('haystack.preview.document_stores.in_memory.document_store.re')
    def test_from_dict(self, mock_regex):
        if False:
            for i in range(10):
                print('nop')
        data = {'type': 'InMemoryDocumentStore', 'init_parameters': {'bm25_tokenization_regex': 'custom_regex', 'bm25_algorithm': 'BM25Plus', 'bm25_parameters': {'key': 'value'}}}
        store = InMemoryDocumentStore.from_dict(data)
        mock_regex.compile.assert_called_with('custom_regex')
        assert store.tokenizer
        assert store.bm25_algorithm.__name__ == 'BM25Plus'
        assert store.bm25_parameters == {'key': 'value'}

    @pytest.mark.unit
    def test_bm25_retrieval(self, docstore: InMemoryDocumentStore):
        if False:
            return 10
        docstore = InMemoryDocumentStore()
        docs = [Document(content='Hello world'), Document(content='Haystack supports multiple languages')]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query='What languages?', top_k=1)
        assert len(results) == 1
        assert results[0].content == 'Haystack supports multiple languages'

    @pytest.mark.unit
    def test_bm25_retrieval_with_empty_document_store(self, docstore: InMemoryDocumentStore, caplog):
        if False:
            while True:
                i = 10
        caplog.set_level(logging.INFO)
        results = docstore.bm25_retrieval(query='How to test this?', top_k=2)
        assert len(results) == 0
        assert 'No documents found for BM25 retrieval. Returning empty list.' in caplog.text

    @pytest.mark.unit
    def test_bm25_retrieval_empty_query(self, docstore: InMemoryDocumentStore):
        if False:
            print('Hello World!')
        docs = [Document(content='Hello world'), Document(content='Haystack supports multiple languages')]
        docstore.write_documents(docs)
        with pytest.raises(ValueError, match='Query should be a non-empty string'):
            docstore.bm25_retrieval(query='', top_k=1)

    @pytest.mark.unit
    def test_bm25_retrieval_with_different_top_k(self, docstore: InMemoryDocumentStore):
        if False:
            return 10
        docs = [Document(content='Hello world'), Document(content='Haystack supports multiple languages'), Document(content='Python is a popular programming language')]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query='languages', top_k=2)
        assert len(results) == 2
        results = docstore.bm25_retrieval(query='languages', top_k=3)
        assert len(results) == 3

    @pytest.mark.unit
    def test_bm25_retrieval_with_two_queries(self, docstore: InMemoryDocumentStore):
        if False:
            for i in range(10):
                print('nop')
        docs = [Document(content='Javascript is a popular programming language'), Document(content='Java is a popular programming language'), Document(content='Python is a popular programming language'), Document(content='Ruby is a popular programming language'), Document(content='PHP is a popular programming language')]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query='Java', top_k=1)
        assert results[0].content == 'Java is a popular programming language'
        results = docstore.bm25_retrieval(query='Python', top_k=1)
        assert results[0].content == 'Python is a popular programming language'

    @pytest.mark.skip(reason='Filter is not working properly, see https://github.com/deepset-ai/haystack/issues/6153')
    def test_eq_filter_embedding(self, docstore: InMemoryDocumentStore, filterable_docs):
        if False:
            for i in range(10):
                print('nop')
        pass

    @pytest.mark.unit
    def test_bm25_retrieval_with_updated_docs(self, docstore: InMemoryDocumentStore):
        if False:
            i = 10
            return i + 15
        docs = [Document(content='Hello world')]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query='Python', top_k=1)
        assert len(results) == 1
        docstore.write_documents([Document(content='Python is a popular programming language')])
        results = docstore.bm25_retrieval(query='Python', top_k=1)
        assert len(results) == 1
        assert results[0].content == 'Python is a popular programming language'
        docstore.write_documents([Document(content='Java is a popular programming language')])
        results = docstore.bm25_retrieval(query='Python', top_k=1)
        assert len(results) == 1
        assert results[0].content == 'Python is a popular programming language'

    @pytest.mark.unit
    def test_bm25_retrieval_with_scale_score(self, docstore: InMemoryDocumentStore):
        if False:
            print('Hello World!')
        docs = [Document(content='Python programming'), Document(content='Java programming')]
        docstore.write_documents(docs)
        results1 = docstore.bm25_retrieval(query='Python', top_k=1, scale_score=True)
        assert results1[0].score is not None
        assert 0.0 <= results1[0].score <= 1.0
        results = docstore.bm25_retrieval(query='Python', top_k=1, scale_score=False)
        assert results[0].score != results1[0].score

    @pytest.mark.unit
    def test_bm25_retrieval_with_table_content(self, docstore: InMemoryDocumentStore):
        if False:
            i = 10
            return i + 15
        table_content = pd.DataFrame({'language': ['Python', 'Java'], 'use': ['Data Science', 'Web Development']})
        docs = [Document(dataframe=table_content), Document(content='Gardening'), Document(content='Bird watching')]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query='Java', top_k=1)
        assert len(results) == 1
        df = results[0].dataframe
        assert isinstance(df, pd.DataFrame)
        assert df.equals(table_content)

    @pytest.mark.unit
    def test_bm25_retrieval_with_text_and_table_content(self, docstore: InMemoryDocumentStore, caplog):
        if False:
            while True:
                i = 10
        table_content = pd.DataFrame({'language': ['Python', 'Java'], 'use': ['Data Science', 'Web Development']})
        document = Document(content='Gardening', dataframe=table_content)
        docs = [document, Document(content='Python'), Document(content='Bird Watching'), Document(content='Gardening'), Document(content='Java')]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query='Gardening', top_k=2)
        assert document in results
        assert 'both text and dataframe content' in caplog.text
        results = docstore.bm25_retrieval(query='Python', top_k=2)
        assert document not in results

    @pytest.mark.unit
    def test_bm25_retrieval_default_filter_for_text_and_dataframes(self, docstore: InMemoryDocumentStore):
        if False:
            for i in range(10):
                print('nop')
        docs = [Document(), Document(content='Gardening'), Document(content='Bird watching')]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query="doesn't matter, top_k is 10", top_k=10)
        assert len(results) == 2

    @pytest.mark.unit
    def test_bm25_retrieval_with_filters(self, docstore: InMemoryDocumentStore):
        if False:
            print('Hello World!')
        selected_document = Document(content='Gardening', meta={'selected': True})
        docs = [Document(), selected_document, Document(content='Bird watching')]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query='Java', top_k=10, filters={'selected': True})
        assert results == [selected_document]

    @pytest.mark.unit
    def test_bm25_retrieval_with_filters_keeps_default_filters(self, docstore: InMemoryDocumentStore):
        if False:
            print('Hello World!')
        docs = [Document(meta={'selected': True}), Document(content='Gardening'), Document(content='Bird watching')]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query='Java', top_k=10, filters={'selected': True})
        assert len(results) == 0

    @pytest.mark.unit
    def test_bm25_retrieval_with_filters_on_text_or_dataframe(self, docstore: InMemoryDocumentStore):
        if False:
            while True:
                i = 10
        document = Document(dataframe=pd.DataFrame({'language': ['Python', 'Java'], 'use': ['Data Science', 'Web']}))
        docs = [Document(), Document(content='Gardening'), Document(content='Bird watching'), document]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query='Java', top_k=10, filters={'content': None})
        assert results == [document]

    @pytest.mark.unit
    def test_bm25_retrieval_with_documents_with_mixed_content(self, docstore: InMemoryDocumentStore):
        if False:
            i = 10
            return i + 15
        double_document = Document(content='Gardening', embedding=[1.0, 2.0, 3.0])
        docs = [Document(embedding=[1.0, 2.0, 3.0]), double_document, Document(content='Bird watching')]
        docstore.write_documents(docs)
        results = docstore.bm25_retrieval(query='Java', top_k=10, filters={'embedding': {'$not': None}})
        assert results == [double_document]

    @pytest.mark.unit
    def test_embedding_retrieval(self):
        if False:
            print('Hello World!')
        docstore = InMemoryDocumentStore(embedding_similarity_function='cosine')
        docs = [Document(content='Hello world', embedding=[0.1, 0.2, 0.3, 0.4]), Document(content='Haystack supports multiple languages', embedding=[1.0, 1.0, 1.0, 1.0])]
        docstore.write_documents(docs)
        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, filters={}, scale_score=False)
        assert len(results) == 1
        assert results[0].content == 'Haystack supports multiple languages'

    @pytest.mark.unit
    def test_embedding_retrieval_invalid_query(self):
        if False:
            return 10
        docstore = InMemoryDocumentStore()
        with pytest.raises(ValueError, match='query_embedding should be a non-empty list of floats'):
            docstore.embedding_retrieval(query_embedding=[])
        with pytest.raises(ValueError, match='query_embedding should be a non-empty list of floats'):
            docstore.embedding_retrieval(query_embedding=['invalid', 'list', 'of', 'strings'])

    @pytest.mark.unit
    def test_embedding_retrieval_no_embeddings(self, caplog):
        if False:
            return 10
        caplog.set_level(logging.WARNING)
        docstore = InMemoryDocumentStore()
        docs = [Document(content='Hello world'), Document(content='Haystack supports multiple languages')]
        docstore.write_documents(docs)
        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1])
        assert len(results) == 0
        assert 'No Documents found with embeddings. Returning empty list.' in caplog.text

    @pytest.mark.unit
    def test_embedding_retrieval_some_documents_wo_embeddings(self, caplog):
        if False:
            for i in range(10):
                print('nop')
        caplog.set_level(logging.INFO)
        docstore = InMemoryDocumentStore()
        docs = [Document(content='Hello world', embedding=[0.1, 0.2, 0.3, 0.4]), Document(content='Haystack supports multiple languages')]
        docstore.write_documents(docs)
        docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1])
        assert "Skipping some Documents that don't have an embedding." in caplog.text

    @pytest.mark.unit
    def test_embedding_retrieval_documents_different_embedding_sizes(self):
        if False:
            while True:
                i = 10
        docstore = InMemoryDocumentStore()
        docs = [Document(content='Hello world', embedding=[0.1, 0.2, 0.3, 0.4]), Document(content='Haystack supports multiple languages', embedding=[1.0, 1.0])]
        docstore.write_documents(docs)
        with pytest.raises(DocumentStoreError, match='The embedding size of all Documents should be the same.'):
            docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1])

    @pytest.mark.unit
    def test_embedding_retrieval_query_documents_different_embedding_sizes(self):
        if False:
            while True:
                i = 10
        docstore = InMemoryDocumentStore()
        docs = [Document(content='Hello world', embedding=[0.1, 0.2, 0.3, 0.4])]
        docstore.write_documents(docs)
        with pytest.raises(DocumentStoreError, match='The embedding size of the query should be the same as the embedding size of the Documents.'):
            docstore.embedding_retrieval(query_embedding=[0.1, 0.1])

    @pytest.mark.unit
    def test_embedding_retrieval_with_different_top_k(self):
        if False:
            return 10
        docstore = InMemoryDocumentStore()
        docs = [Document(content='Hello world', embedding=[0.1, 0.2, 0.3, 0.4]), Document(content='Haystack supports multiple languages', embedding=[1.0, 1.0, 1.0, 1.0]), Document(content='Python is a popular programming language', embedding=[0.5, 0.5, 0.5, 0.5])]
        docstore.write_documents(docs)
        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2)
        assert len(results) == 2
        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=3)
        assert len(results) == 3

    @pytest.mark.unit
    def test_embedding_retrieval_with_scale_score(self):
        if False:
            return 10
        docstore = InMemoryDocumentStore()
        docs = [Document(content='Hello world', embedding=[0.1, 0.2, 0.3, 0.4]), Document(content='Haystack supports multiple languages', embedding=[1.0, 1.0, 1.0, 1.0]), Document(content='Python is a popular programming language', embedding=[0.5, 0.5, 0.5, 0.5])]
        docstore.write_documents(docs)
        results1 = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, scale_score=True)
        assert results1[0].score is not None
        assert 0.0 <= results1[0].score <= 1.0
        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, scale_score=False)
        assert results[0].score != results1[0].score

    @pytest.mark.unit
    def test_embedding_retrieval_return_embedding(self):
        if False:
            return 10
        docstore = InMemoryDocumentStore(embedding_similarity_function='cosine')
        docs = [Document(content='Hello world', embedding=[0.1, 0.2, 0.3, 0.4]), Document(content='Haystack supports multiple languages', embedding=[1.0, 1.0, 1.0, 1.0])]
        docstore.write_documents(docs)
        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, return_embedding=False)
        assert results[0].embedding is None
        results = docstore.embedding_retrieval(query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, return_embedding=True)
        assert results[0].embedding == [1.0, 1.0, 1.0, 1.0]

    @pytest.mark.unit
    def test_compute_cosine_similarity_scores(self):
        if False:
            return 10
        docstore = InMemoryDocumentStore(embedding_similarity_function='cosine')
        docs = [Document(content='Document 1', embedding=[1.0, 0.0, 0.0, 0.0]), Document(content='Document 2', embedding=[1.0, 1.0, 1.0, 1.0])]
        scores = docstore._compute_query_embedding_similarity_scores(embedding=[0.1, 0.1, 0.1, 0.1], documents=docs, scale_score=False)
        assert scores == [0.5, 1.0]

    @pytest.mark.unit
    def test_compute_dot_product_similarity_scores(self):
        if False:
            print('Hello World!')
        docstore = InMemoryDocumentStore(embedding_similarity_function='dot_product')
        docs = [Document(content='Document 1', embedding=[1.0, 0.0, 0.0, 0.0]), Document(content='Document 2', embedding=[1.0, 1.0, 1.0, 1.0])]
        scores = docstore._compute_query_embedding_similarity_scores(embedding=[0.1, 0.1, 0.1, 0.1], documents=docs, scale_score=False)
        assert scores == [0.1, 0.4]