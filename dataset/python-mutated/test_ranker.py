import pytest
import math
import warnings
import logging
import copy
from unittest.mock import patch
import torch
from haystack.schema import Document
from haystack.nodes.ranker.base import BaseRanker
from haystack.nodes.ranker import SentenceTransformersRanker, CohereRanker
from haystack.nodes.ranker.recentness_ranker import RecentnessRanker
from haystack.errors import HaystackError, NodeError

@pytest.fixture
def docs():
    if False:
        i = 10
        return i + 15
    docs = [Document(content='Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother\'s spokesman (""prophet"") to the Pharaoh. Part of the Law (Torah) that Moses received from', meta={'name': '0'}, id='1'), Document(content='Democratic Republic of the Congo to the south. Angola\'s capital, Luanda, lies on the Atlantic coast in the northwest of the country. Angola, although located in a tropical zone, has a climate that is not characterized for this region, due to the confluence of three factors: As a result, Angola\'s climate is characterized by two seasons: rainfall from October to April and drought, known as ""Cacimbo"", from May to August, drier, as the name implies, and with lower temperatures. On the other hand, while the coastline has high rainfall rates, decreasing from North to South and from to , with', id='2'), Document(content='Schopenhauer, describing him as an ultimately shallow thinker: ""Schopenhauer has quite a crude mind ... where real depth starts, his comes to an end."" His friend Bertrand Russell had a low opinion on the philosopher, and attacked him in his famous ""History of Western Philosophy"" for hypocritically praising asceticism yet not acting upon it. On the opposite isle of Russell on the foundations of mathematics, the Dutch mathematician L. E. J. Brouwer incorporated the ideas of Kant and Schopenhauer in intuitionism, where mathematics is considered a purely mental activity, instead of an analytic activity wherein objective properties of reality are', meta={'name': '1'}, id='3'), Document(content='The Dothraki vocabulary was created by David J. Peterson well in advance of the adaptation. HBO hired the Language Creatio', meta={'name': '2'}, id='4'), Document(content="The title of the episode refers to the Great Sept of Baelor, the main religious building in King's Landing, where the episode's pivotal scene takes place. In the world created by George R. R. Martin", meta={}, id='5')]
    return docs

@pytest.fixture
def mock_cohere_post():
    if False:
        for i in range(10):
            print('nop')

    class Response:

        def __init__(self, text: str):
            if False:
                return 10
            self.text = text
    with patch('haystack.nodes.ranker.cohere.CohereRanker._post') as cohere_post:
        cohere_post.return_value = Response(text='{"id":"73701fd4-fe30-4007-9698-e960a51b19b4","results":[{"index":4,"relevance_score":0.9937345},{"index":3,"relevance_score":0.2232077},{"index":0,"relevance_score":0.006538825},{"index":2,"relevance_score":0.002278331},{"index":1,"relevance_score":0.000035633544}],"meta":{"api_version":{"version":"1"}}}')
        yield cohere_post

@pytest.fixture
def mock_transformer_tokenizer():
    if False:
        return 10

    class Features(dict):

        def to(self, arg):
            if False:
                print('Hello World!')
            return self

    class Tokenizer:

        def __call__(self, *args, **kwargs):
            if False:
                print('Hello World!')
            return Features({'input_ids': torch.zeros([5, 162]), 'token_type_ids': torch.zeros([5, 162], dtype=torch.long), 'attention_mask': torch.zeros([5, 162], dtype=torch.long)})
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
        mock_tokenizer.return_value = Tokenizer()
        yield mock_tokenizer

@pytest.fixture
def mock_transformer_model():
    if False:
        return 10

    class Logits:

        def __init__(self, logits):
            if False:
                while True:
                    i = 10
            self.logits = logits

    class Model:

        def __init__(self):
            if False:
                return 10
            self.logits = torch.tensor([[-9.7414], [-11.1572], [-11.1708], [-11.1515], [5.2571]])
            self.num_labels = 1

        def __call__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            return Logits(logits=self.logits)

        def eval(self):
            if False:
                i = 10
                return i + 15
            return self

        def to(self, arg):
            if False:
                i = 10
                return i + 15
            return self
    with patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model:
        mock_model.return_value = Model()
        yield mock_model

@pytest.mark.unit
def test_ranker_preprocess_batch_queries_and_docs_raises():
    if False:
        print('Hello World!')
    query_1 = 'query 1'
    query_2 = 'query 2'
    docs = [Document(content='dummy doc 1')]
    with patch('haystack.nodes.ranker.sentence_transformers.SentenceTransformersRanker.__init__') as mock_ranker_init:
        mock_ranker_init.return_value = None
        ranker = SentenceTransformersRanker(model_name_or_path='fake_model')
    with pytest.raises(HaystackError, match='Number of queries must be 1 if a single list of Documents is provided.'):
        (_, _, _, _) = ranker._preprocess_batch_queries_and_docs(queries=[query_1, query_2], documents=docs)

@pytest.mark.unit
def test_ranker_preprocess_batch_queries_and_docs_single_query_single_doc_list():
    if False:
        return 10
    query1 = 'query 1'
    docs1 = [Document(content='dummy doc 1'), Document(content='dummy doc 2')]
    with patch('haystack.nodes.ranker.sentence_transformers.SentenceTransformersRanker.__init__') as mock_ranker_init:
        mock_ranker_init.return_value = None
        ranker = SentenceTransformersRanker(model_name_or_path='fake_model')
    (num_of_docs, all_queries, all_docs, single_list_of_docs) = ranker._preprocess_batch_queries_and_docs(queries=[query1], documents=docs1)
    assert single_list_of_docs is True
    assert num_of_docs == [2]
    assert len(all_queries) == 2
    assert len(all_docs) == 2

@pytest.mark.unit
def test_ranker_preprocess_batch_queries_and_docs_multiple_queries_multiple_doc_lists():
    if False:
        for i in range(10):
            print('nop')
    query_1 = 'query 1'
    query_2 = 'query 2'
    docs1 = [Document(content='dummy doc 1'), Document(content='dummy doc 2')]
    docs2 = [Document(content='dummy doc 3')]
    with patch('haystack.nodes.ranker.sentence_transformers.SentenceTransformersRanker.__init__') as mock_ranker_init:
        mock_ranker_init.return_value = None
        ranker = SentenceTransformersRanker(model_name_or_path='fake_model')
    (num_of_docs, all_queries, all_docs, single_list_of_docs) = ranker._preprocess_batch_queries_and_docs(queries=[query_1, query_2], documents=[docs1, docs2])
    assert single_list_of_docs is False
    assert num_of_docs == [2, 1]
    assert len(all_queries) == 3
    assert len(all_docs) == 3

@pytest.mark.unit
def test_ranker_get_batches():
    if False:
        return 10
    all_queries = ['query 1', 'query 1']
    all_docs = [Document(content='dummy doc 1'), Document(content='dummy doc 2')]
    batches = SentenceTransformersRanker._get_batches(all_queries=all_queries, all_docs=all_docs, batch_size=None)
    assert next(batches) == (all_queries, all_docs)
    batches = SentenceTransformersRanker._get_batches(all_queries=all_queries, all_docs=all_docs, batch_size=1)
    assert next(batches) == (all_queries[0:1], all_docs[0:1])

@pytest.mark.unit
def test_add_meta_fields_to_docs():
    if False:
        return 10
    docs = [Document(content='dummy doc 1', meta={'str_field': 'test1', 'empty_str_field': '', 'numeric_field': 2.0, 'list_field': ['item0.1', 'item0.2'], 'empty_list_field': []}), Document(content='dummy doc 2', meta={'str_field': 'test2', 'empty_str_field': '', 'numeric_field': 5.0, 'list_field': ['item1.1', 'item1.2'], 'empty_list_field': []})]
    with patch('haystack.nodes.ranker.sentence_transformers.SentenceTransformersRanker.__init__') as mock_ranker_init:
        mock_ranker_init.return_value = None
        ranker = SentenceTransformersRanker(model_name_or_path='fake_model')
    docs_with_meta = ranker._add_meta_fields_to_docs(documents=docs, embed_meta_fields=['str_field', 'empty_str_field', 'numeric_field', 'list_field', 'empty_list_field'])
    assert docs_with_meta[0].content.startswith('test1\n2.0\nitem0.1\nitem0.2\ndummy doc 1')
    assert docs_with_meta[1].content.startswith('test2\n5.0\nitem1.1\nitem1.2\ndummy doc 2')

@pytest.mark.unit
def test_add_meta_fields_to_docs_none():
    if False:
        print('Hello World!')
    docs = [Document(content='dummy doc 1', meta={'none_field': None})]
    with patch('haystack.nodes.ranker.sentence_transformers.SentenceTransformersRanker.__init__') as mock_ranker_init:
        mock_ranker_init.return_value = None
        ranker = SentenceTransformersRanker(model_name_or_path='fake_model')
    docs_with_meta = ranker._add_meta_fields_to_docs(documents=docs, embed_meta_fields=['none_field'])
    assert docs_with_meta == docs

@pytest.mark.unit
def test_add_meta_fields_to_docs_non_existent():
    if False:
        while True:
            i = 10
    docs = [Document(content='dummy doc 1', meta={'test_field': 'A string'})]
    with patch('haystack.nodes.ranker.sentence_transformers.SentenceTransformersRanker.__init__') as mock_ranker_init:
        mock_ranker_init.return_value = None
        ranker = SentenceTransformersRanker(model_name_or_path='fake_model')
    docs_with_meta = ranker._add_meta_fields_to_docs(documents=docs, embed_meta_fields=['wrong_field'])
    assert docs_with_meta == docs

@pytest.mark.unit
def test_add_meta_fields_to_docs_empty_list():
    if False:
        i = 10
        return i + 15
    docs = [Document(content='dummy doc 1', meta={'test_field': 'A string'})]
    with patch('haystack.nodes.ranker.sentence_transformers.SentenceTransformersRanker.__init__') as mock_ranker_init:
        mock_ranker_init.return_value = None
        ranker = SentenceTransformersRanker(model_name_or_path='fake_model')
    docs_with_meta = ranker._add_meta_fields_to_docs(documents=docs, embed_meta_fields=[])
    assert docs_with_meta == docs

@pytest.mark.unit
def test_ranker(docs, mock_transformer_model, mock_transformer_tokenizer):
    if False:
        print('Hello World!')
    with patch('torch.nn.DataParallel'):
        ranker = SentenceTransformersRanker(model_name_or_path='fake_model')
    query = "What is the most important building in King's Landing that has a religious background?"
    results = ranker.predict(query=query, documents=docs)
    assert results[0] == docs[4]

@pytest.mark.unit
def test_ranker_run(docs, mock_transformer_model, mock_transformer_tokenizer):
    if False:
        while True:
            i = 10
    with patch('torch.nn.DataParallel'):
        ranker = SentenceTransformersRanker(model_name_or_path='fake_model')
    query = "What is the most important building in King's Landing that has a religious background?"
    results = ranker.run(query=query, documents=docs)
    assert results[0]['documents'][0] == docs[4]

@pytest.mark.unit
def test_ranker_batch_single_query_single_doc_list(docs, mock_transformer_model, mock_transformer_tokenizer):
    if False:
        i = 10
        return i + 15
    with patch('torch.nn.DataParallel'):
        ranker = SentenceTransformersRanker(model_name_or_path='fake_model')
    query = "What is the most important building in King's Landing that has a religious background?"
    results = ranker.predict_batch(queries=[query], documents=docs)
    assert results[0] == docs[4]

@pytest.mark.unit
def test_ranker_batch_single_query_multiple_doc_lists(docs, mock_transformer_model, mock_transformer_tokenizer):
    if False:
        for i in range(10):
            print('nop')
    with patch('torch.nn.DataParallel'):
        ranker = SentenceTransformersRanker(model_name_or_path='fake_model', batch_size=5)
    query = "What is the most important building in King's Landing that has a religious background?"
    results = ranker.predict_batch(queries=[query], documents=[docs, docs])
    assert isinstance(results, list)
    assert isinstance(results[0], list)
    for reranked_docs in results:
        assert reranked_docs[0] == docs[4]

def test_ranker_batch_multiple_queries_multiple_doc_lists(ranker, docs):
    if False:
        while True:
            i = 10
    query_1 = "What is the most important building in King's Landing that has a religious background?"
    query_2 = "How is Angola's climate characterized?"
    results = ranker.predict_batch(queries=[query_1, query_2], documents=[docs, docs])
    assert isinstance(results, list)
    assert isinstance(results[0], list)
    assert results[0][0] == docs[4]
    assert results[1][0] == docs[1]

@pytest.mark.unit
def test_ranker_with_embed_meta_fields(docs, mock_transformer_model, mock_transformer_tokenizer):
    if False:
        i = 10
        return i + 15
    with patch('torch.nn.DataParallel'):
        ranker = SentenceTransformersRanker(model_name_or_path='fake_model', embed_meta_fields=['name'])
    query = "What is the most important building in King's Landing that has a religious background?"
    results = ranker.predict(query=query, documents=docs)
    assert results[0] == docs[4]

def test_ranker_two_logits(ranker_two_logits, docs):
    if False:
        print('Hello World!')
    assert isinstance(ranker_two_logits, BaseRanker)
    assert isinstance(ranker_two_logits, SentenceTransformersRanker)
    query = 'Welches ist das wichtigste Gebäude in Königsmund, das einen religiösen Hintergrund hat?'
    docs = [Document(content='Aaron Aaron (oder ; "Ahärôn") ist ein Prophet, Hohepriester und der Bruder von Moses in den abrahamitischen Religionen. Aaron ist ebenso wie sein Bruder Moses ausschließlich aus religiösen Texten wie der Bibel und dem Koran bekannt. Die hebräische Bibel berichtet, dass Aaron und seine ältere Schwester Mirjam im Gegensatz zu Mose, der am ägyptischen Königshof aufwuchs, bei ihren Verwandten im östlichen Grenzland Ägyptens (Goschen) blieben. Als Mose den ägyptischen König zum ersten Mal mit den Israeliten konfrontierte, fungierte Aaron als Sprecher ("Prophet") seines Bruders gegenüber dem Pharao. Ein Teil des Gesetzes (Tora), das Mose von', meta={'name': '0'}, id='1'), Document(content='Demokratische Republik Kongo im Süden. Die angolanische Hauptstadt Luanda liegt an der Atlantikküste im Nordwesten des Landes. Angola liegt zwar in einer tropischen Zone, hat aber ein Klima, das aufgrund des Zusammenwirkens von drei Faktoren nicht für diese Region typisch ist: So ist das Klima Angolas durch zwei Jahreszeiten gekennzeichnet: Regenfälle von Oktober bis April und die als "Cacimbo" bezeichnete Dürre von Mai bis August, die, wie der Name schon sagt, trockener ist und niedrigere Temperaturen aufweist. Andererseits sind die Niederschlagsmengen an der Küste sehr hoch und nehmen von Norden nach Süden und von Süden nach Süden ab, mit', id='2'), Document(content='Schopenhauer, indem er ihn als einen letztlich oberflächlichen Denker beschreibt: ""Schopenhauer hat einen ziemlich groben Verstand ... wo wirkliche Tiefe beginnt, hört seine auf."" Sein Freund Bertrand Russell hatte eine schlechte Meinung von dem Philosophen und griff ihn in seiner berühmten "Geschichte der westlichen Philosophie" an, weil er heuchlerisch die Askese lobte, aber nicht danach handelte. Der holländische Mathematiker L. E. J. Brouwer, der auf der gegenüberliegenden Insel von Russell über die Grundlagen der Mathematik sprach, nahm die Ideen von Kant und Schopenhauer in den Intuitionismus auf, in dem die Mathematik als eine rein geistige Tätigkeit betrachtet wird und nicht als eine analytische Tätigkeit, bei der die objektiven Eigenschaften der Realität berücksichtigt werden.', meta={'name': '1'}, id='3'), Document(content='Das dothrakische Vokabular wurde von David J. Peterson lange vor der Verfilmung erstellt. HBO beauftragte das Language Creatio', meta={'name': '2'}, id='4'), Document(content='Der Titel der Episode bezieht sich auf die Große Septe von Baelor, das wichtigste religiöse Gebäude in Königsmund, in dem die Schlüsselszene der Episode stattfindet. In der von George R. R. Martin geschaffenen Welt', meta={}, id='5')]
    results = ranker_two_logits.predict(query=query, documents=docs)
    assert results[0] == docs[4]

def test_ranker_returns_normalized_score(ranker):
    if False:
        return 10
    query = "What is the most important building in King's Landing that has a religious background?"
    docs = [Document(content='Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother\'s spokesman (""prophet"") to the Pharaoh. Part of the Law (Torah) that Moses received from', meta={'name': '0'}, id='1')]
    results = ranker.predict(query=query, documents=docs)
    score = results[0].score
    precomputed_score = 5.8796231e-05
    assert math.isclose(precomputed_score, score, rel_tol=0.01)

def test_ranker_returns_raw_score_when_no_scaling():
    if False:
        i = 10
        return i + 15
    ranker = SentenceTransformersRanker(model_name_or_path='cross-encoder/ms-marco-MiniLM-L-12-v2', scale_score=False)
    query = "What is the most important building in King's Landing that has a religious background?"
    docs = [Document(content='Aaron Aaron ( or ; ""Ahärôn"") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother\'s spokesman (""prophet"") to the Pharaoh. Part of the Law (Torah) that Moses received from', meta={'name': '0'}, id='1')]
    results = ranker.predict(query=query, documents=docs)
    score = results[0].score
    precomputed_score = -9.744687
    assert math.isclose(precomputed_score, score, rel_tol=0.001)

def test_ranker_returns_raw_score_for_two_logits(ranker_two_logits):
    if False:
        for i in range(10):
            print('nop')
    query = 'Welches ist das wichtigste Gebäude in Königsmund, das einen religiösen Hintergrund hat?'
    docs = [Document(content='Aaron Aaron (oder ; "Ahärôn") ist ein Prophet, Hohepriester und der Bruder von Moses in den abrahamitischen Religionen. Aaron ist ebenso wie sein Bruder Moses ausschließlich aus religiösen Texten wie der Bibel und dem Koran bekannt. Die hebräische Bibel berichtet, dass Aaron und seine ältere Schwester Mirjam im Gegensatz zu Mose, der am ägyptischen Königshof aufwuchs, bei ihren Verwandten im östlichen Grenzland Ägyptens (Goschen) blieben. Als Mose den ägyptischen König zum ersten Mal mit den Israeliten konfrontierte, fungierte Aaron als Sprecher ("Prophet") seines Bruders gegenüber dem Pharao. Ein Teil des Gesetzes (Tora), das Mose von', meta={'name': '0'}, id='1')]
    results = ranker_two_logits.predict(query=query, documents=docs)
    score = results[0].score
    precomputed_score = -3.61354
    assert math.isclose(precomputed_score, score, rel_tol=0.001)

def test_predict_batch_returns_correct_number_of_docs(ranker):
    if False:
        i = 10
        return i + 15
    docs = [Document(content=f'test {number}') for number in range(5)]
    assert len(ranker.predict('where is test 3?', docs, top_k=4)) == 4
    assert len(ranker.predict_batch(['where is test 3?'], docs, batch_size=2, top_k=4)) == 4

@pytest.mark.unit
def test_cohere_ranker(docs, mock_cohere_post):
    if False:
        while True:
            i = 10
    query = "What is the most important building in King's Landing that has a religious background?"
    ranker = CohereRanker(api_key='fake_key', model_name_or_path='rerank-english-v2.0')
    results = ranker.predict(query=query, documents=docs)
    mock_cohere_post.assert_called_once_with({'model': 'rerank-english-v2.0', 'query': query, 'documents': [{'text': d.content} for d in docs], 'top_n': None, 'return_documents': False, 'max_chunks_per_doc': None})
    assert results[0] == docs[4]

@pytest.mark.unit
def test_cohere_ranker_with_embed_meta_fields(docs, mock_cohere_post):
    if False:
        for i in range(10):
            print('nop')
    query = "What is the most important building in King's Landing that has a religious background?"
    ranker = CohereRanker(api_key='fake_key', model_name_or_path='rerank-english-v2.0', embed_meta_fields=['name'])
    results = ranker.predict(query=query, documents=docs)
    documents = []
    for d in docs:
        meta = d.meta.get('name')
        if meta:
            documents.append({'text': d.meta['name'] + '\n' + d.content})
        else:
            documents.append({'text': d.content})
    mock_cohere_post.assert_called_once_with({'model': 'rerank-english-v2.0', 'query': query, 'documents': documents, 'top_n': None, 'return_documents': False, 'max_chunks_per_doc': None})
    assert results[0] == docs[4]

@pytest.mark.unit
def test_cohere_ranker_batch_single_query_single_doc_list(docs, mock_cohere_post):
    if False:
        while True:
            i = 10
    query = "What is the most important building in King's Landing that has a religious background?"
    ranker = CohereRanker(api_key='fake_key', model_name_or_path='rerank-english-v2.0')
    results = ranker.predict_batch(queries=[query], documents=docs)
    mock_cohere_post.assert_called_once_with({'model': 'rerank-english-v2.0', 'query': query, 'documents': [{'text': d.content} for d in docs], 'top_n': None, 'return_documents': False, 'max_chunks_per_doc': None})
    assert results[0] == docs[4]

@pytest.mark.unit
def test_cohere_ranker_batch_single_query_single_doc_list_with_embed_meta_fields(docs, mock_cohere_post):
    if False:
        print('Hello World!')
    query = "What is the most important building in King's Landing that has a religious background?"
    ranker = CohereRanker(api_key='fake_key', model_name_or_path='rerank-english-v2.0', embed_meta_fields=['name'])
    results = ranker.predict_batch(queries=[query], documents=docs)
    documents = []
    for d in docs:
        meta = d.meta.get('name')
        if meta:
            documents.append({'text': d.meta['name'] + '\n' + d.content})
        else:
            documents.append({'text': d.content})
    mock_cohere_post.assert_called_once_with({'model': 'rerank-english-v2.0', 'query': query, 'documents': documents, 'top_n': None, 'return_documents': False, 'max_chunks_per_doc': None})
    assert results[0] == docs[4]

@pytest.mark.unit
def test_cohere_ranker_batch_single_query_multiple_doc_lists(docs, mock_cohere_post):
    if False:
        return 10
    query = "What is the most important building in King's Landing that has a religious background?"
    ranker = CohereRanker(api_key='fake_key', model_name_or_path='rerank-english-v2.0')
    results = ranker.predict_batch(queries=[query], documents=[docs, docs])
    assert mock_cohere_post.call_count == 2
    mock_cohere_post.assert_called_with({'model': 'rerank-english-v2.0', 'query': query, 'documents': [{'text': d.content} for d in docs], 'top_n': None, 'return_documents': False, 'max_chunks_per_doc': None})
    assert isinstance(results, list)
    assert isinstance(results[0], list)
    for reranked_docs in results:
        assert reranked_docs[0] == docs[4]

@pytest.mark.unit
def test_cohere_ranker_batch_multiple_queries_multiple_doc_lists(docs, mock_cohere_post):
    if False:
        while True:
            i = 10
    query = "What is the most important building in King's Landing that has a religious background?"
    ranker = CohereRanker(api_key='fake_key', model_name_or_path='rerank-english-v2.0')
    results = ranker.predict_batch(queries=[query, query], documents=[docs, docs])
    assert mock_cohere_post.call_count == 2
    mock_cohere_post.assert_called_with({'model': 'rerank-english-v2.0', 'query': query, 'documents': [{'text': d.content} for d in docs], 'top_n': None, 'return_documents': False, 'max_chunks_per_doc': None})
    assert isinstance(results, list)
    assert isinstance(results[0], list)
    assert results[0][0] == docs[4]
    assert results[1][0] == docs[4]
recency_tests_inputs = [pytest.param({'docs': [{'meta': {'date': '2021-02-11'}, 'score': 0.3, 'id': '1'}, {'meta': {'date': '2024-02-11'}, 'score': 0.4, 'id': '2'}, {'meta': {'date': '2020-02-11'}, 'score': 0.6, 'id': '3'}], 'weight': 0.5, 'date_meta_field': 'date', 'top_k': 2, 'ranking_mode': 'score', 'expected_scores': {'1': 0.4833333333333333, '2': 0.7}, 'expected_order': ['2', '1'], 'expected_logs': [], 'expected_warning': ''}, id='Score ranking mode works as expected'), pytest.param({'docs': [{'meta': {'date': '2021-02-11'}, 'id': '1'}, {'meta': {'date': '2018-02-11'}, 'id': '2'}, {'meta': {'date': '2020-02-11'}, 'id': '3'}], 'weight': 0.5, 'date_meta_field': 'date', 'top_k': 2, 'ranking_mode': 'reciprocal_rank_fusion', 'expected_scores': {'1': 0.01639344262295082, '2': 0.016001024065540194}, 'expected_order': ['1', '2'], 'expected_logs': [], 'expected_warning': ''}, id='RRF ranking mode works as expected'), pytest.param({'docs': [{'meta': {'data': '2021-02-11'}, 'score': 0.3, 'id': '1'}, {'meta': {'date': '2024-02-11'}, 'score': 0.4, 'id': '2'}, {'meta': {'date': '2020-02-11'}, 'score': 0.6, 'id': '3'}], 'weight': 0.5, 'date_meta_field': 'date', 'expected_scores': {'1': 0.3, '2': 0.4, '3': 0.6}, 'expected_order': ['1', '2', '3'], 'expected_exception': NodeError("\n                Param <date_meta_field> was set to 'date', but document(s) 1 do not contain this metadata key.\n\n                Please double-check the names of existing metadata fields of your documents \n\n                and set <date_meta_field> to the name of the field that contains dates.\n                "), 'top_k': 2, 'ranking_mode': 'score'}, id='Wrong field to find the date'), pytest.param({'docs': [{'meta': {'date': 'abcd'}, 'id': '1'}, {'meta': {'date': '2024-02-11'}, 'id': '2'}, {'meta': {'date': '2020-02-11'}, 'id': '3'}], 'weight': 0.5, 'date_meta_field': 'date', 'expected_order': ['1', '2', '3'], 'expected_logs': [('haystack.nodes.ranker.recentness_ranker', logging.ERROR, '\n                Could not parse date information for dates: abcd - 2024-02-11 - 2020-02-11\n\n                Continuing without sorting by date.\n                ')], 'top_k': 2, 'ranking_mode': 'reciprocal_rank_fusion'}, id='Date unparsable'), pytest.param({'docs': [{'meta': {'date': '2021-02-11'}, 'score': 1.3, 'id': '1'}, {'meta': {'date': '2024-02-11'}, 'score': 0.4, 'id': '2'}, {'meta': {'date': '2020-02-11'}, 'score': 0.6, 'id': '3'}], 'weight': 0.5, 'date_meta_field': 'date', 'top_k': 2, 'ranking_mode': 'score', 'expected_scores': {'1': 0.5, '2': 0.7, '3': 0.4666666666666667}, 'expected_order': ['2', '3'], 'expected_warning': ['The score 1.3 for document 1 is outside the [0,1] range; defaulting to 0']}, id='Wrong score, outside of bonds'), pytest.param({'docs': [{'meta': {'date': '2021-02-11'}, 'id': '1'}, {'meta': {'date': '2024-02-11'}, 'score': 0.4, 'id': '2'}, {'meta': {'date': '2020-02-11'}, 'score': 0.6, 'id': '3'}], 'weight': 0.5, 'date_meta_field': 'date', 'top_k': 2, 'ranking_mode': 'score', 'expected_scores': {'1': 0.5, '2': 0.7, '3': 0.4666666666666667}, 'expected_order': ['2', '3'], 'expected_warning': ['The score was not provided; defaulting to 0']}, id='Wrong score, not provided'), pytest.param({'docs': [{'meta': {'date': '2021-02-11'}, 'id': '1'}, {'meta': {'date': '2024-02-11'}, 'score': 0.4, 'id': '2'}, {'meta': {'date': '2020-02-11'}, 'score': 0.6, 'id': '3'}], 'weight': 0.5, 'date_meta_field': 'date', 'top_k': 2, 'ranking_mode': 'blablabla', 'expected_scores': {'1': 0.01626123744050767, '2': 0.01626123744050767}, 'expected_order': ['1', '2'], 'expected_exception': NodeError("\n                Param <ranking_mode> needs to be 'reciprocal_rank_fusion' or 'score' but was set to 'blablabla'. \n\n                Please change the <ranking_mode> when initializing the RecentnessRanker.\n                ")}, id='Wrong ranking mode provided')]

@pytest.mark.unit
@pytest.mark.parametrize('test_input', recency_tests_inputs)
def test_recentness_ranker(caplog, test_input):
    if False:
        i = 10
        return i + 15
    docs = []
    for doc in test_input['docs']:
        docs.append(Document(content='abc', **doc))
    with warnings.catch_warnings(record=True) as warnings_list:
        ranker = RecentnessRanker(date_meta_field=test_input['date_meta_field'], ranking_mode=test_input['ranking_mode'], weight=test_input['weight'])
        predict_exception = None
        results = []
        try:
            results = ranker.predict(query='', documents=docs, top_k=test_input['top_k'])
        except Exception as e:
            predict_exception = e
        check_results(results, test_input, warnings_list, caplog, predict_exception)

@pytest.mark.unit
@pytest.mark.parametrize('test_input', recency_tests_inputs)
def test_recentness_ranker_batch_list(caplog, test_input):
    if False:
        while True:
            i = 10
    docs = []
    for doc in test_input['docs']:
        docs.append(Document(content='abc', **doc))
    with warnings.catch_warnings(record=True) as warnings_list:
        ranker = RecentnessRanker(date_meta_field=test_input['date_meta_field'], ranking_mode=test_input['ranking_mode'], weight=test_input['weight'])
        predict_exception = None
        results = []
        try:
            results = ranker.predict_batch(queries='', documents=docs, top_k=test_input['top_k'])
        except Exception as e:
            predict_exception = e
        check_results(results, test_input, warnings_list, caplog, predict_exception)

@pytest.mark.unit
@pytest.mark.parametrize('test_input', recency_tests_inputs)
def test_recentness_ranker_batch_list_of_lists(caplog, test_input):
    if False:
        while True:
            i = 10
    docs = []
    for doc in test_input['docs']:
        docs.append(Document(content='abc', **doc))
    with warnings.catch_warnings(record=True) as warnings_list:
        ranker = RecentnessRanker(date_meta_field=test_input['date_meta_field'], ranking_mode=test_input['ranking_mode'], weight=test_input['weight'])
        predict_exception = None
        results = []
        try:
            results = ranker.predict_batch(queries='', documents=[docs, copy.deepcopy(docs)], top_k=test_input['top_k'])
        except Exception as e:
            predict_exception = e
        check_results(results, test_input, warnings_list, caplog, predict_exception, list_of_lists=True)

def check_results(results, test_input, warnings_list, caplog, exception, list_of_lists=False):
    if False:
        print('Hello World!')
    expected_logs_count = 1
    if list_of_lists:
        expected_logs_count = 2
    if 'expected_exception' in test_input and test_input['expected_exception'] is not None:
        assert exception.message == test_input['expected_exception'].message
        assert type(exception) == type(test_input['expected_exception'])
        return
    else:
        assert exception is None
    if 'expected_warning' not in test_input or test_input['expected_warning'] == []:
        assert len(warnings_list) == 0
    else:
        assert len(warnings_list) == len(test_input['expected_warning'])
        for i in range(len(warnings_list)):
            assert test_input['expected_warning'][int(i)] == str(warnings_list[i].message)
    if 'expected_logs' not in test_input or test_input['expected_logs'] == []:
        assert len(caplog.record_tuples) == 0
    else:
        assert expected_logs_count * len(test_input['expected_logs']) == len(caplog.record_tuples)
        for i in range(len(caplog.record_tuples)):
            assert test_input['expected_logs'][int(i / expected_logs_count)] == caplog.record_tuples[i]
    if not list_of_lists:
        check_result_content(results, test_input)
    else:
        for i in results:
            check_result_content(i, test_input)

def check_result_content(results, test_input):
    if False:
        for i in range(10):
            print('nop')
    assert len(results) == len(test_input['expected_order'])
    for i in range(len(test_input['expected_order'])):
        assert test_input['expected_order'][i] == results[i].id
        if 'expected_scores' in test_input:
            assert test_input['expected_scores'][results[i].id] == results[i].score
        else:
            assert results[i].score is None