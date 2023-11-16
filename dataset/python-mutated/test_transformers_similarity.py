import pytest
from haystack.preview import Document, ComponentError
from haystack.preview.components.rankers.transformers_similarity import TransformersSimilarityRanker

class TestSimilarityRanker:

    @pytest.mark.unit
    def test_to_dict(self):
        if False:
            i = 10
            return i + 15
        component = TransformersSimilarityRanker()
        data = component.to_dict()
        assert data == {'type': 'TransformersSimilarityRanker', 'init_parameters': {'device': 'cpu', 'top_k': 10, 'token': None, 'model_name_or_path': 'cross-encoder/ms-marco-MiniLM-L-6-v2'}}

    @pytest.mark.unit
    def test_to_dict_with_custom_init_parameters(self):
        if False:
            i = 10
            return i + 15
        component = TransformersSimilarityRanker(model_name_or_path='my_model', device='cuda', token='my_token', top_k=5)
        data = component.to_dict()
        assert data == {'type': 'TransformersSimilarityRanker', 'init_parameters': {'device': 'cuda', 'model_name_or_path': 'my_model', 'token': None, 'top_k': 5}}

    @pytest.mark.integration
    @pytest.mark.parametrize('query,docs_before_texts,expected_first_text', [('City in Bosnia and Herzegovina', ['Berlin', 'Belgrade', 'Sarajevo'], 'Sarajevo'), ('Machine learning', ['Python', 'Bakery in Paris', 'Tesla Giga Berlin'], 'Python'), ('Cubist movement', ['Nirvana', 'Pablo Picasso', 'Coffee'], 'Pablo Picasso')])
    def test_run(self, query, docs_before_texts, expected_first_text):
        if False:
            print('Hello World!')
        '\n        Test if the component ranks documents correctly.\n        '
        ranker = TransformersSimilarityRanker(model_name_or_path='cross-encoder/ms-marco-MiniLM-L-6-v2')
        ranker.warm_up()
        docs_before = [Document(content=text) for text in docs_before_texts]
        output = ranker.run(query=query, documents=docs_before)
        docs_after = output['documents']
        assert len(docs_after) == 3
        assert docs_after[0].content == expected_first_text
        sorted_scores = sorted([doc.score for doc in docs_after], reverse=True)
        assert [doc.score for doc in docs_after] == sorted_scores

    @pytest.mark.integration
    def test_returns_empty_list_if_no_documents_are_provided(self):
        if False:
            for i in range(10):
                print('nop')
        sampler = TransformersSimilarityRanker()
        sampler.warm_up()
        output = sampler.run(query='City in Germany', documents=[])
        assert not output['documents']

    @pytest.mark.integration
    def test_raises_component_error_if_model_not_warmed_up(self):
        if False:
            while True:
                i = 10
        sampler = TransformersSimilarityRanker()
        with pytest.raises(ComponentError):
            sampler.run(query='query', documents=[Document(content='document')])

    @pytest.mark.integration
    @pytest.mark.parametrize('query,docs_before_texts,expected_first_text', [('City in Bosnia and Herzegovina', ['Berlin', 'Belgrade', 'Sarajevo'], 'Sarajevo'), ('Machine learning', ['Python', 'Bakery in Paris', 'Tesla Giga Berlin'], 'Python'), ('Cubist movement', ['Nirvana', 'Pablo Picasso', 'Coffee'], 'Pablo Picasso')])
    def test_run_top_k(self, query, docs_before_texts, expected_first_text):
        if False:
            print('Hello World!')
        '\n        Test if the component ranks documents correctly with a custom top_k.\n        '
        ranker = TransformersSimilarityRanker(model_name_or_path='cross-encoder/ms-marco-MiniLM-L-6-v2', top_k=2)
        ranker.warm_up()
        docs_before = [Document(content=text) for text in docs_before_texts]
        output = ranker.run(query=query, documents=docs_before)
        docs_after = output['documents']
        assert len(docs_after) == 2
        assert docs_after[0].content == expected_first_text
        sorted_scores = sorted([doc.score for doc in docs_after], reverse=True)
        assert [doc.score for doc in docs_after] == sorted_scores