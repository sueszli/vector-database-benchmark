import random
import pytest
from haystack.preview import Document, ComponentError
from haystack.preview.components.samplers.top_p import TopPSampler

class TestTopPSampler:

    @pytest.mark.unit
    def test_run_scores_from_metadata(self):
        if False:
            return 10
        '\n        Test if the component runs correctly with scores already in the metadata.\n        '
        sampler = TopPSampler(top_p=0.95, score_field='similarity_score')
        docs = [Document(content='Berlin', meta={'similarity_score': -10.6}), Document(content='Belgrade', meta={'similarity_score': -8.9}), Document(content='Sarajevo', meta={'similarity_score': -4.6})]
        output = sampler.run(documents=docs)
        docs = output['documents']
        assert len(docs) == 1
        assert docs[0].content == 'Sarajevo'

    @pytest.mark.unit
    def test_run_scores(self):
        if False:
            i = 10
            return i + 15
        '\n        Test if the component runs correctly with scores in the Document score field.\n        '
        sampler = TopPSampler(top_p=0.99)
        docs = [Document(content='Berlin', score=-10.6), Document(content='Belgrade', score=-8.9), Document(content='Sarajevo', score=-4.6)]
        random.shuffle(docs)
        sorted_scores = sorted([doc.score for doc in docs], reverse=True)
        output = sampler.run(documents=docs)
        docs_filtered = output['documents']
        assert len(docs_filtered) == 1
        assert docs_filtered[0].content == 'Sarajevo'
        assert [doc.score for doc in docs_filtered] == sorted_scores[:1]

    @pytest.mark.unit
    def test_run_scores_top_p_1(self):
        if False:
            return 10
        '\n        Test if the component runs correctly top_p=1.\n        '
        sampler = TopPSampler(top_p=1.0)
        docs = [Document(content='Berlin', score=-10.6), Document(content='Belgrade', score=-8.9), Document(content='Sarajevo', score=-4.6)]
        random.shuffle(docs)
        output = sampler.run(documents=docs)
        docs_filtered = output['documents']
        assert len(docs_filtered) == len(docs)
        assert docs_filtered[0].content == 'Sarajevo'
        assert [doc.score for doc in docs_filtered] == sorted([doc.score for doc in docs], reverse=True)

    @pytest.mark.unit
    def test_returns_empty_list_if_no_documents_are_provided(self):
        if False:
            return 10
        sampler = TopPSampler()
        output = sampler.run(documents=[])
        assert output['documents'] == []

    @pytest.mark.unit
    def test_run_scores_no_metadata_present(self):
        if False:
            i = 10
            return i + 15
        '\n        Test if the component runs correctly with scores missing from the metadata yet being specified in the\n        score_field.\n        '
        sampler = TopPSampler(top_p=0.95, score_field='similarity_score')
        docs = [Document(content='Berlin', score=-10.6), Document(content='Belgrade', score=-8.9), Document(content='Sarajevo', score=-4.6)]
        with pytest.raises(ComponentError, match="Score field 'similarity_score' not found"):
            sampler.run(documents=docs)