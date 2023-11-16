import logging
from unittest.mock import patch
import pytest
import sys
from copy import deepcopy
from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes.preprocessor import PreProcessor
from haystack.nodes.query_classifier.transformers import TransformersQueryClassifier
from haystack.nodes.retriever.dense import DensePassageRetriever
from haystack.nodes.retriever.sparse import BM25Retriever
from haystack.nodes.summarizer.transformers import TransformersSummarizer
from haystack.pipelines.base import Pipeline
from haystack.pipelines import ExtractiveQAPipeline, GenerativeQAPipeline, SearchSummarizationPipeline
from haystack.pipelines.standard_pipelines import DocumentSearchPipeline, FAQPipeline, RetrieverQuestionGenerationPipeline, TranslationWrapperPipeline
from haystack.nodes.translator.transformers import TransformersTranslator
from haystack.schema import Answer, Document, EvaluationResult, Label, MultiLabel, Span

@pytest.mark.unit
@patch('haystack.pipelines.base.Pipeline.run_batch')
def test_eval_batch_add_isolated_node_eval_passed_to_run_batch(mock_run_batch):
    if False:
        while True:
            i = 10
    pipeline = Pipeline()
    pipeline.eval_batch(labels=EVAL_LABELS, add_isolated_node_eval=True)
    (_, kwargs) = mock_run_batch.call_args
    assert 'add_isolated_node_eval' in kwargs['params']
    assert kwargs['params']['add_isolated_node_eval'] is True

@pytest.mark.skipif(sys.platform in ['win32', 'cygwin'], reason='Causes OOM on windows github runner')
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
@pytest.mark.parametrize('retriever_with_docs', ['embedding'], indirect=True)
def test_summarizer_calculate_metrics(document_store_with_docs: ElasticsearchDocumentStore, retriever_with_docs):
    if False:
        i = 10
        return i + 15
    document_store_with_docs.update_embeddings(retriever=retriever_with_docs)
    summarizer = TransformersSummarizer(model_name_or_path='sshleifer/distill-pegasus-xsum-16-4', use_gpu=False)
    pipeline = SearchSummarizationPipeline(retriever=retriever_with_docs, summarizer=summarizer, return_in_answer_format=True)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=EVAL_LABELS, params={'Retriever': {'top_k': 5}}, context_matching_min_length=10)
    metrics = eval_result.calculate_metrics(document_scope='context')
    assert 'Retriever' in eval_result
    assert 'Summarizer' in eval_result
    assert len(eval_result) == 2
    assert metrics['Retriever']['mrr'] == 1.0
    assert metrics['Retriever']['map'] == 1.0
    assert metrics['Retriever']['recall_multi_hit'] == 1.0
    assert metrics['Retriever']['recall_single_hit'] == 1.0
    assert metrics['Retriever']['precision'] == 1.0
    assert metrics['Retriever']['ndcg'] == 1.0
    assert metrics['Summarizer']['mrr'] == 1.0
    assert metrics['Summarizer']['map'] == 1.0
    assert metrics['Summarizer']['recall_multi_hit'] == 1.0
    assert metrics['Summarizer']['recall_single_hit'] == 1.0
    assert metrics['Summarizer']['precision'] == 1.0
    assert metrics['Summarizer']['ndcg'] == 1.0
EVAL_LABELS = [MultiLabel(labels=[Label(query='Who lives in Berlin?', answer=Answer(answer='Carla', offsets_in_context=[Span(11, 16)]), document=Document(id='a0747b83aea0b60c4b114b15476dd32d', content_type='text', content='My name is Carla and I live in Berlin'), is_correct_answer=True, is_correct_document=True, origin='gold-label')]), MultiLabel(labels=[Label(query='Who lives in Munich?', answer=Answer(answer='Carla', offsets_in_context=[Span(11, 16)]), document=Document(id='something_else', content_type='text', content='My name is Carla and I live in Munich'), is_correct_answer=True, is_correct_document=True, origin='gold-label')])]

@pytest.mark.parametrize('retriever_with_docs', ['tfidf'], indirect=True)
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
@pytest.mark.parametrize('reader', ['farm', 'transformers'], indirect=True)
def test_extractive_qa_eval(reader, retriever_with_docs, tmp_path):
    if False:
        print('Hello World!')
    labels = EVAL_LABELS[:1]
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result = pipeline.eval_batch(labels=labels, params={'Retriever': {'top_k': 5}})
    metrics = eval_result.calculate_metrics(document_scope='document_id')
    reader_result = eval_result['Reader']
    retriever_result = eval_result['Retriever']
    assert reader_result[reader_result['rank'] == 1]['answer'].iloc[0] in reader_result[reader_result['rank'] == 1]['gold_answers'].iloc[0]
    assert retriever_result[retriever_result['rank'] == 1]['document_id'].iloc[0] in retriever_result[retriever_result['rank'] == 1]['gold_document_ids'].iloc[0]
    assert metrics['Reader']['exact_match'] == 1.0
    assert metrics['Reader']['f1'] == 1.0
    assert metrics['Retriever']['mrr'] == 1.0
    assert metrics['Retriever']['recall_multi_hit'] == 1.0
    assert metrics['Retriever']['recall_single_hit'] == 1.0
    assert metrics['Retriever']['precision'] == 0.2
    assert metrics['Retriever']['map'] == 1.0
    assert metrics['Retriever']['ndcg'] == 1.0
    eval_result.save(tmp_path)
    saved_eval_result = EvaluationResult.load(tmp_path)
    metrics = saved_eval_result.calculate_metrics(document_scope='document_id')
    assert reader_result[reader_result['rank'] == 1]['answer'].iloc[0] in reader_result[reader_result['rank'] == 1]['gold_answers'].iloc[0]
    assert retriever_result[retriever_result['rank'] == 1]['document_id'].iloc[0] in retriever_result[retriever_result['rank'] == 1]['gold_document_ids'].iloc[0]
    assert metrics['Reader']['exact_match'] == 1.0
    assert metrics['Reader']['f1'] == 1.0
    assert metrics['Retriever']['mrr'] == 1.0
    assert metrics['Retriever']['recall_multi_hit'] == 1.0
    assert metrics['Retriever']['recall_single_hit'] == 1.0
    assert metrics['Retriever']['precision'] == 0.2
    assert metrics['Retriever']['map'] == 1.0
    assert metrics['Retriever']['ndcg'] == 1.0

@pytest.mark.parametrize('retriever_with_docs', ['tfidf'], indirect=True)
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
@pytest.mark.parametrize('reader', ['farm'], indirect=True)
def test_extractive_qa_eval_multiple_queries(reader, retriever_with_docs, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=EVAL_LABELS, params={'Retriever': {'top_k': 5}})
    metrics = eval_result.calculate_metrics(document_scope='document_id')
    reader_result = eval_result['Reader']
    retriever_result = eval_result['Retriever']
    reader_berlin = reader_result[reader_result['query'] == 'Who lives in Berlin?']
    reader_munich = reader_result[reader_result['query'] == 'Who lives in Munich?']
    retriever_berlin = retriever_result[retriever_result['query'] == 'Who lives in Berlin?']
    retriever_munich = retriever_result[retriever_result['query'] == 'Who lives in Munich?']
    assert reader_berlin[reader_berlin['rank'] == 1]['answer'].iloc[0] in reader_berlin[reader_berlin['rank'] == 1]['gold_answers'].iloc[0]
    assert retriever_berlin[retriever_berlin['rank'] == 1]['document_id'].iloc[0] in retriever_berlin[retriever_berlin['rank'] == 1]['gold_document_ids'].iloc[0]
    assert reader_munich[reader_munich['rank'] == 1]['answer'].iloc[0] not in reader_munich[reader_munich['rank'] == 1]['gold_answers'].iloc[0]
    assert retriever_munich[retriever_munich['rank'] == 1]['document_id'].iloc[0] not in retriever_munich[retriever_munich['rank'] == 1]['gold_document_ids'].iloc[0]
    assert metrics['Reader']['exact_match'] == 1.0
    assert metrics['Reader']['f1'] == 1.0
    assert metrics['Retriever']['mrr'] == 0.5
    assert metrics['Retriever']['map'] == 0.5
    assert metrics['Retriever']['recall_multi_hit'] == 0.5
    assert metrics['Retriever']['recall_single_hit'] == 0.5
    assert metrics['Retriever']['precision'] == 0.1
    assert metrics['Retriever']['ndcg'] == 0.5
    eval_result.save(tmp_path)
    saved_eval_result = EvaluationResult.load(tmp_path)
    metrics = saved_eval_result.calculate_metrics(document_scope='document_id')
    assert reader_berlin[reader_berlin['rank'] == 1]['answer'].iloc[0] in reader_berlin[reader_berlin['rank'] == 1]['gold_answers'].iloc[0]
    assert retriever_berlin[retriever_berlin['rank'] == 1]['document_id'].iloc[0] in retriever_berlin[retriever_berlin['rank'] == 1]['gold_document_ids'].iloc[0]
    assert reader_munich[reader_munich['rank'] == 1]['answer'].iloc[0] not in reader_munich[reader_munich['rank'] == 1]['gold_answers'].iloc[0]
    assert retriever_munich[retriever_munich['rank'] == 1]['document_id'].iloc[0] not in retriever_munich[retriever_munich['rank'] == 1]['gold_document_ids'].iloc[0]
    assert metrics['Reader']['exact_match'] == 1.0
    assert metrics['Reader']['f1'] == 1.0
    assert metrics['Retriever']['mrr'] == 0.5
    assert metrics['Retriever']['map'] == 0.5
    assert metrics['Retriever']['recall_multi_hit'] == 0.5
    assert metrics['Retriever']['recall_single_hit'] == 0.5
    assert metrics['Retriever']['precision'] == 0.1
    assert metrics['Retriever']['ndcg'] == 0.5

@pytest.mark.parametrize('retriever_with_docs', ['tfidf'], indirect=True)
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
@pytest.mark.parametrize('reader', ['farm'], indirect=True)
def test_extractive_qa_eval_sas(reader, retriever_with_docs):
    if False:
        i = 10
        return i + 15
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=EVAL_LABELS, params={'Retriever': {'top_k': 5}}, sas_model_name_or_path='sentence-transformers/paraphrase-MiniLM-L3-v2')
    metrics = eval_result.calculate_metrics(document_scope='document_id')
    assert metrics['Reader']['exact_match'] == 1.0
    assert metrics['Reader']['f1'] == 1.0
    assert metrics['Retriever']['mrr'] == 0.5
    assert metrics['Retriever']['map'] == 0.5
    assert metrics['Retriever']['recall_multi_hit'] == 0.5
    assert metrics['Retriever']['recall_single_hit'] == 0.5
    assert metrics['Retriever']['precision'] == 0.1
    assert metrics['Retriever']['ndcg'] == 0.5
    assert 'sas' in metrics['Reader']
    assert metrics['Reader']['sas'] == pytest.approx(1.0)

@pytest.mark.parametrize('reader', ['farm'], indirect=True)
def test_reader_eval_in_pipeline(reader):
    if False:
        i = 10
        return i + 15
    pipeline = Pipeline()
    pipeline.add_node(component=reader, name='Reader', inputs=['Query'])
    eval_result: EvaluationResult = pipeline.eval_batch(labels=EVAL_LABELS, documents=[[label.document for label in multilabel.labels] for multilabel in EVAL_LABELS], params={})
    metrics = eval_result.calculate_metrics(document_scope='document_id')
    assert metrics['Reader']['exact_match'] == 1.0
    assert metrics['Reader']['f1'] == 1.0

@pytest.mark.parametrize('retriever_with_docs', ['tfidf'], indirect=True)
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
def test_extractive_qa_eval_document_scope(retriever_with_docs):
    if False:
        return 10
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=EVAL_LABELS, params={'Retriever': {'top_k': 5}}, context_matching_min_length=20)
    metrics = eval_result.calculate_metrics(document_scope='document_id')
    assert metrics['Retriever']['mrr'] == 0.5
    assert metrics['Retriever']['map'] == 0.5
    assert metrics['Retriever']['recall_multi_hit'] == 0.5
    assert metrics['Retriever']['recall_single_hit'] == 0.5
    assert metrics['Retriever']['precision'] == 0.1
    assert metrics['Retriever']['ndcg'] == 0.5
    metrics = eval_result.calculate_metrics(document_scope='context')
    assert metrics['Retriever']['mrr'] == 1.0
    assert metrics['Retriever']['map'] == 1.0
    assert metrics['Retriever']['recall_multi_hit'] == 1.0
    assert metrics['Retriever']['recall_single_hit'] == 1.0
    assert metrics['Retriever']['precision'] == 1.0
    assert metrics['Retriever']['ndcg'] == 1.0
    metrics = eval_result.calculate_metrics(document_scope='document_id_and_context')
    assert metrics['Retriever']['mrr'] == 0.5
    assert metrics['Retriever']['map'] == 0.5
    assert metrics['Retriever']['recall_multi_hit'] == 0.5
    assert metrics['Retriever']['recall_single_hit'] == 0.5
    assert metrics['Retriever']['precision'] == 0.1
    assert metrics['Retriever']['ndcg'] == 0.5
    metrics = eval_result.calculate_metrics(document_scope='document_id_or_context')
    assert metrics['Retriever']['mrr'] == 1.0
    assert metrics['Retriever']['map'] == 1.0
    assert metrics['Retriever']['recall_multi_hit'] == 1.0
    assert metrics['Retriever']['recall_single_hit'] == 1.0
    assert metrics['Retriever']['precision'] == 1.0
    assert metrics['Retriever']['ndcg'] == 1.0
    metrics = eval_result.calculate_metrics(document_scope='answer')
    assert metrics['Retriever']['mrr'] == 1.0
    assert metrics['Retriever']['map'] == 1.0
    assert metrics['Retriever']['recall_multi_hit'] == 1.0
    assert metrics['Retriever']['recall_single_hit'] == 1.0
    assert metrics['Retriever']['precision'] == 0.2
    assert metrics['Retriever']['ndcg'] == 1.0
    metrics = eval_result.calculate_metrics(document_scope='document_id_or_answer')
    assert metrics['Retriever']['mrr'] == 1.0
    assert metrics['Retriever']['map'] == 1.0
    assert metrics['Retriever']['recall_multi_hit'] == 1.0
    assert metrics['Retriever']['recall_single_hit'] == 1.0
    assert metrics['Retriever']['precision'] == 0.2
    assert metrics['Retriever']['ndcg'] == 1.0

@pytest.mark.parametrize('retriever_with_docs', ['tfidf'], indirect=True)
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
@pytest.mark.parametrize('reader', ['farm'], indirect=True)
def test_extractive_qa_eval_answer_scope(reader, retriever_with_docs):
    if False:
        i = 10
        return i + 15
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=EVAL_LABELS, params={'Retriever': {'top_k': 5}}, sas_model_name_or_path='sentence-transformers/paraphrase-MiniLM-L3-v2', context_matching_min_length=20)
    metrics = eval_result.calculate_metrics(answer_scope='any')
    assert metrics['Retriever']['mrr'] == 1.0
    assert metrics['Retriever']['map'] == 1.0
    assert metrics['Retriever']['recall_multi_hit'] == 1.0
    assert metrics['Retriever']['recall_single_hit'] == 1.0
    assert metrics['Retriever']['precision'] == 0.2
    assert metrics['Retriever']['ndcg'] == 1.0
    assert metrics['Reader']['exact_match'] == 1.0
    assert metrics['Reader']['f1'] == 1.0
    assert metrics['Reader']['sas'] == pytest.approx(1.0)
    metrics = eval_result.calculate_metrics(answer_scope='context')
    assert metrics['Retriever']['mrr'] == 1.0
    assert metrics['Retriever']['map'] == 1.0
    assert metrics['Retriever']['recall_multi_hit'] == 1.0
    assert metrics['Retriever']['recall_single_hit'] == 1.0
    assert metrics['Retriever']['precision'] == 0.2
    assert metrics['Retriever']['ndcg'] == 1.0
    assert metrics['Reader']['exact_match'] == 1.0
    assert metrics['Reader']['f1'] == 1.0
    assert metrics['Reader']['sas'] == pytest.approx(1.0)
    metrics = eval_result.calculate_metrics(answer_scope='document_id')
    assert metrics['Retriever']['mrr'] == 0.5
    assert metrics['Retriever']['map'] == 0.5
    assert metrics['Retriever']['recall_multi_hit'] == 0.5
    assert metrics['Retriever']['recall_single_hit'] == 0.5
    assert metrics['Retriever']['precision'] == 0.1
    assert metrics['Retriever']['ndcg'] == 0.5
    assert metrics['Reader']['exact_match'] == 0.5
    assert metrics['Reader']['f1'] == 0.5
    assert metrics['Reader']['sas'] == pytest.approx(0.5)
    metrics = eval_result.calculate_metrics(answer_scope='document_id_and_context')
    assert metrics['Retriever']['mrr'] == 0.5
    assert metrics['Retriever']['map'] == 0.5
    assert metrics['Retriever']['recall_multi_hit'] == 0.5
    assert metrics['Retriever']['recall_single_hit'] == 0.5
    assert metrics['Retriever']['precision'] == 0.1
    assert metrics['Retriever']['ndcg'] == 0.5
    assert metrics['Reader']['exact_match'] == 0.5
    assert metrics['Reader']['f1'] == 0.5
    assert metrics['Reader']['sas'] == pytest.approx(0.5)

@pytest.mark.parametrize('retriever_with_docs', ['tfidf'], indirect=True)
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
@pytest.mark.parametrize('reader', ['farm'], indirect=True)
def test_extractive_qa_eval_answer_document_scope_combinations(reader, retriever_with_docs, caplog):
    if False:
        return 10
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=EVAL_LABELS, params={'Retriever': {'top_k': 5}}, sas_model_name_or_path='sentence-transformers/paraphrase-MiniLM-L3-v2', context_matching_min_length=20)
    with caplog.at_level(logging.WARNING):
        eval_result.calculate_metrics(document_scope='document_id_or_answer', answer_scope='context')
        eval_result.calculate_metrics(document_scope='answer', answer_scope='context')
        assert 'You specified a non-answer document_scope together with a non-default answer_scope' not in caplog.text
    with caplog.at_level(logging.WARNING):
        eval_result.calculate_metrics(document_scope='document_id', answer_scope='context')
        assert 'You specified a non-answer document_scope together with a non-default answer_scope' in caplog.text
    with caplog.at_level(logging.WARNING):
        eval_result.calculate_metrics(document_scope='context', answer_scope='context')
        assert 'You specified a non-answer document_scope together with a non-default answer_scope' in caplog.text
    with caplog.at_level(logging.WARNING):
        eval_result.calculate_metrics(document_scope='document_id_and_context', answer_scope='context')
        assert 'You specified a non-answer document_scope together with a non-default answer_scope' in caplog.text
    with caplog.at_level(logging.WARNING):
        eval_result.calculate_metrics(document_scope='document_id_or_context', answer_scope='context')
        assert 'You specified a non-answer document_scope together with a non-default answer_scope' in caplog.text

@pytest.mark.parametrize('retriever_with_docs', ['tfidf'], indirect=True)
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
@pytest.mark.parametrize('reader', ['farm'], indirect=True)
def test_extractive_qa_eval_simulated_top_k_reader(reader, retriever_with_docs):
    if False:
        for i in range(10):
            print('nop')
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=EVAL_LABELS, params={'Retriever': {'top_k': 5}}, sas_model_name_or_path='sentence-transformers/paraphrase-MiniLM-L3-v2')
    metrics_top_1 = eval_result.calculate_metrics(simulated_top_k_reader=1, document_scope='document_id')
    assert metrics_top_1['Reader']['exact_match'] == 0.5
    assert metrics_top_1['Reader']['f1'] == 0.5
    assert metrics_top_1['Reader']['sas'] == pytest.approx(0.6003, abs=0.0001)
    assert metrics_top_1['Retriever']['mrr'] == 0.5
    assert metrics_top_1['Retriever']['map'] == 0.5
    assert metrics_top_1['Retriever']['recall_multi_hit'] == 0.5
    assert metrics_top_1['Retriever']['recall_single_hit'] == 0.5
    assert metrics_top_1['Retriever']['precision'] == 0.1
    assert metrics_top_1['Retriever']['ndcg'] == 0.5
    metrics_top_2 = eval_result.calculate_metrics(simulated_top_k_reader=2, document_scope='document_id')
    assert metrics_top_2['Reader']['exact_match'] == 0.5
    assert metrics_top_2['Reader']['f1'] == 0.5
    assert metrics_top_2['Reader']['sas'] == pytest.approx(0.6003, abs=0.0001)
    assert metrics_top_2['Retriever']['mrr'] == 0.5
    assert metrics_top_2['Retriever']['map'] == 0.5
    assert metrics_top_2['Retriever']['recall_multi_hit'] == 0.5
    assert metrics_top_2['Retriever']['recall_single_hit'] == 0.5
    assert metrics_top_2['Retriever']['precision'] == 0.1
    assert metrics_top_2['Retriever']['ndcg'] == 0.5
    metrics_top_5 = eval_result.calculate_metrics(simulated_top_k_reader=5, document_scope='document_id')
    assert metrics_top_5['Reader']['exact_match'] == 1.0
    assert metrics_top_5['Reader']['f1'] == 1.0
    assert metrics_top_5['Reader']['sas'] == pytest.approx(1.0, abs=0.0001)
    assert metrics_top_5['Retriever']['mrr'] == 0.5
    assert metrics_top_5['Retriever']['map'] == 0.5
    assert metrics_top_5['Retriever']['recall_multi_hit'] == 0.5
    assert metrics_top_5['Retriever']['recall_single_hit'] == 0.5
    assert metrics_top_5['Retriever']['precision'] == 0.1
    assert metrics_top_5['Retriever']['ndcg'] == 0.5

@pytest.mark.parametrize('retriever_with_docs', ['tfidf'], indirect=True)
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
@pytest.mark.parametrize('reader', ['farm'], indirect=True)
def test_extractive_qa_eval_simulated_top_k_retriever(reader, retriever_with_docs):
    if False:
        print('Hello World!')
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=EVAL_LABELS, params={'Retriever': {'top_k': 5}})
    metrics_top_10 = eval_result.calculate_metrics(document_scope='document_id')
    assert metrics_top_10['Reader']['exact_match'] == 1.0
    assert metrics_top_10['Reader']['f1'] == 1.0
    assert metrics_top_10['Retriever']['mrr'] == 0.5
    assert metrics_top_10['Retriever']['map'] == 0.5
    assert metrics_top_10['Retriever']['recall_multi_hit'] == 0.5
    assert metrics_top_10['Retriever']['recall_single_hit'] == 0.5
    assert metrics_top_10['Retriever']['precision'] == 0.1
    assert metrics_top_10['Retriever']['ndcg'] == 0.5
    metrics_top_1 = eval_result.calculate_metrics(simulated_top_k_retriever=1, document_scope='document_id')
    assert metrics_top_1['Reader']['exact_match'] == 1.0
    assert metrics_top_1['Reader']['f1'] == 1.0
    assert metrics_top_1['Retriever']['mrr'] == 0.5
    assert metrics_top_1['Retriever']['map'] == 0.5
    assert metrics_top_1['Retriever']['recall_multi_hit'] == 0.5
    assert metrics_top_1['Retriever']['recall_single_hit'] == 0.5
    assert metrics_top_1['Retriever']['precision'] == 0.5
    assert metrics_top_1['Retriever']['ndcg'] == 0.5
    metrics_top_2 = eval_result.calculate_metrics(simulated_top_k_retriever=2, document_scope='document_id')
    assert metrics_top_2['Reader']['exact_match'] == 1.0
    assert metrics_top_2['Reader']['f1'] == 1.0
    assert metrics_top_2['Retriever']['mrr'] == 0.5
    assert metrics_top_2['Retriever']['map'] == 0.5
    assert metrics_top_2['Retriever']['recall_multi_hit'] == 0.5
    assert metrics_top_2['Retriever']['recall_single_hit'] == 0.5
    assert metrics_top_2['Retriever']['precision'] == 0.25
    assert metrics_top_2['Retriever']['ndcg'] == 0.5
    metrics_top_3 = eval_result.calculate_metrics(simulated_top_k_retriever=3, document_scope='document_id')
    assert metrics_top_3['Reader']['exact_match'] == 1.0
    assert metrics_top_3['Reader']['f1'] == 1.0
    assert metrics_top_3['Retriever']['mrr'] == 0.5
    assert metrics_top_3['Retriever']['map'] == 0.5
    assert metrics_top_3['Retriever']['recall_multi_hit'] == 0.5
    assert metrics_top_3['Retriever']['recall_single_hit'] == 0.5
    assert metrics_top_3['Retriever']['precision'] == 1.0 / 6
    assert metrics_top_3['Retriever']['ndcg'] == 0.5

@pytest.mark.parametrize('retriever_with_docs', ['tfidf'], indirect=True)
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
@pytest.mark.parametrize('reader', ['farm'], indirect=True)
def test_extractive_qa_eval_simulated_top_k_reader_and_retriever(reader, retriever_with_docs):
    if False:
        print('Hello World!')
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=EVAL_LABELS, params={'Retriever': {'top_k': 10}})
    metrics_top_10 = eval_result.calculate_metrics(simulated_top_k_reader=1, document_scope='document_id')
    assert metrics_top_10['Reader']['exact_match'] == 0.5
    assert metrics_top_10['Reader']['f1'] == 0.5
    assert metrics_top_10['Retriever']['mrr'] == 0.5
    assert metrics_top_10['Retriever']['map'] == 0.5
    assert metrics_top_10['Retriever']['recall_multi_hit'] == 0.5
    assert metrics_top_10['Retriever']['recall_single_hit'] == 0.5
    assert metrics_top_10['Retriever']['precision'] == 0.1
    assert metrics_top_10['Retriever']['ndcg'] == 0.5
    metrics_top_1 = eval_result.calculate_metrics(simulated_top_k_reader=1, simulated_top_k_retriever=1, document_scope='document_id')
    assert metrics_top_1['Reader']['exact_match'] == 1.0
    assert metrics_top_1['Reader']['f1'] == 1.0
    assert metrics_top_1['Retriever']['mrr'] == 0.5
    assert metrics_top_1['Retriever']['map'] == 0.5
    assert metrics_top_1['Retriever']['recall_multi_hit'] == 0.5
    assert metrics_top_1['Retriever']['recall_single_hit'] == 0.5
    assert metrics_top_1['Retriever']['precision'] == 0.5
    assert metrics_top_1['Retriever']['ndcg'] == 0.5
    metrics_top_2 = eval_result.calculate_metrics(simulated_top_k_reader=1, simulated_top_k_retriever=2, document_scope='document_id')
    assert metrics_top_2['Reader']['exact_match'] == 0.5
    assert metrics_top_2['Reader']['f1'] == 0.5
    assert metrics_top_2['Retriever']['mrr'] == 0.5
    assert metrics_top_2['Retriever']['map'] == 0.5
    assert metrics_top_2['Retriever']['recall_multi_hit'] == 0.5
    assert metrics_top_2['Retriever']['recall_single_hit'] == 0.5
    assert metrics_top_2['Retriever']['precision'] == 0.25
    assert metrics_top_2['Retriever']['ndcg'] == 0.5
    metrics_top_3 = eval_result.calculate_metrics(simulated_top_k_reader=1, simulated_top_k_retriever=3, document_scope='document_id')
    assert metrics_top_3['Reader']['exact_match'] == 0.5
    assert metrics_top_3['Reader']['f1'] == 0.5
    assert metrics_top_3['Retriever']['mrr'] == 0.5
    assert metrics_top_3['Retriever']['map'] == 0.5
    assert metrics_top_3['Retriever']['recall_multi_hit'] == 0.5
    assert metrics_top_3['Retriever']['recall_single_hit'] == 0.5
    assert metrics_top_3['Retriever']['precision'] == 1.0 / 6
    assert metrics_top_3['Retriever']['ndcg'] == 0.5

@pytest.mark.parametrize('retriever_with_docs', ['tfidf'], indirect=True)
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
@pytest.mark.parametrize('reader', ['farm'], indirect=True)
def test_extractive_qa_eval_isolated(reader, retriever_with_docs):
    if False:
        i = 10
        return i + 15
    labels = deepcopy(EVAL_LABELS)
    label_copy = deepcopy(labels[0].labels[0])
    label_copy.answer = Answer(answer='I', offsets_in_context=[Span(21, 22)])
    labels[0].labels.append(label_copy)
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=labels, sas_model_name_or_path='sentence-transformers/paraphrase-MiniLM-L3-v2', add_isolated_node_eval=True)
    metrics_top_1 = eval_result.calculate_metrics(simulated_top_k_reader=1, document_scope='document_id')
    assert metrics_top_1['Reader']['exact_match'] == 0.5
    assert metrics_top_1['Reader']['f1'] == 0.5
    assert metrics_top_1['Reader']['sas'] == pytest.approx(0.6003, abs=0.0001)
    assert metrics_top_1['Retriever']['mrr'] == 0.5
    assert metrics_top_1['Retriever']['map'] == 0.5
    assert metrics_top_1['Retriever']['recall_multi_hit'] == 0.5
    assert metrics_top_1['Retriever']['recall_single_hit'] == 0.5
    assert metrics_top_1['Retriever']['precision'] == 1.0 / 10
    assert metrics_top_1['Retriever']['ndcg'] == 0.5
    metrics_top_1 = eval_result.calculate_metrics(simulated_top_k_reader=1, eval_mode='isolated')
    assert metrics_top_1['Reader']['exact_match'] == 1.0
    assert metrics_top_1['Reader']['f1'] == 1.0
    assert metrics_top_1['Reader']['sas'] == pytest.approx(1.0, abs=0.0001)
    assert labels[0].labels[0].id == labels[0].labels[1].id
    reader_eval_df = eval_result.node_results['Reader']
    isolated_reader_eval_df = reader_eval_df[reader_eval_df['eval_mode'] == 'isolated']
    assert len(isolated_reader_eval_df) == len(labels) * reader.top_k_per_candidate

@pytest.mark.parametrize('retriever_with_docs', ['tfidf'], indirect=True)
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
@pytest.mark.parametrize('reader', ['farm'], indirect=True)
def test_extractive_qa_eval_wrong_examples(reader, retriever_with_docs):
    if False:
        for i in range(10):
            print('nop')
    labels = [MultiLabel(labels=[Label(query='Who lives in Berlin?', answer=Answer(answer='Carla', offsets_in_context=[Span(11, 16)]), document=Document(id='a0747b83aea0b60c4b114b15476dd32d', content_type='text', content='My name is Carla and I live in Berlin'), is_correct_answer=True, is_correct_document=True, origin='gold-label')]), MultiLabel(labels=[Label(query='Who lives in Munich?', answer=Answer(answer='Pete', offsets_in_context=[Span(11, 16)]), document=Document(id='something_else', content_type='text', content='My name is Pete and I live in Munich'), is_correct_answer=True, is_correct_document=True, origin='gold-label')])]
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=labels, params={'Retriever': {'top_k': 5}})
    wrongs_retriever = eval_result.wrong_examples(node='Retriever', n=1)
    wrongs_reader = eval_result.wrong_examples(node='Reader', n=1)
    assert len(wrongs_retriever) == 1
    assert len(wrongs_reader) == 1

@pytest.mark.parametrize('retriever_with_docs', ['tfidf'], indirect=True)
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
@pytest.mark.parametrize('reader', ['farm'], indirect=True)
def test_extractive_qa_print_eval_report(reader, retriever_with_docs):
    if False:
        print('Hello World!')
    labels = [MultiLabel(labels=[Label(query='Who lives in Berlin?', answer=Answer(answer='Carla', offsets_in_context=[Span(11, 16)]), document=Document(id='a0747b83aea0b60c4b114b15476dd32d', content_type='text', content='My name is Carla and I live in Berlin'), is_correct_answer=True, is_correct_document=True, origin='gold-label')]), MultiLabel(labels=[Label(query='Who lives in Munich?', answer=Answer(answer='Pete', offsets_in_context=[Span(11, 16)]), document=Document(id='something_else', content_type='text', content='My name is Pete and I live in Munich'), is_correct_answer=True, is_correct_document=True, origin='gold-label')])]
    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=labels, params={'Retriever': {'top_k': 5}})
    pipeline.print_eval_report(eval_result)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=labels, params={'Retriever': {'top_k': 5}}, add_isolated_node_eval=True)
    pipeline.print_eval_report(eval_result)

@pytest.mark.parametrize('retriever_with_docs', ['tfidf'], indirect=True)
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
def test_document_search_calculate_metrics(retriever_with_docs):
    if False:
        i = 10
        return i + 15
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=EVAL_LABELS, params={'Retriever': {'top_k': 5}})
    metrics = eval_result.calculate_metrics(document_scope='document_id')
    assert 'Retriever' in eval_result
    assert len(eval_result) == 1
    retriever_result = eval_result['Retriever']
    retriever_berlin = retriever_result[retriever_result['query'] == 'Who lives in Berlin?']
    retriever_munich = retriever_result[retriever_result['query'] == 'Who lives in Munich?']
    assert retriever_berlin[retriever_berlin['rank'] == 1]['document_id'].iloc[0] in retriever_berlin[retriever_berlin['rank'] == 1]['gold_document_ids'].iloc[0]
    assert retriever_munich[retriever_munich['rank'] == 1]['document_id'].iloc[0] not in retriever_munich[retriever_munich['rank'] == 1]['gold_document_ids'].iloc[0]
    assert metrics['Retriever']['mrr'] == 0.5
    assert metrics['Retriever']['map'] == 0.5
    assert metrics['Retriever']['recall_multi_hit'] == 0.5
    assert metrics['Retriever']['recall_single_hit'] == 0.5
    assert metrics['Retriever']['precision'] == 0.1
    assert metrics['Retriever']['ndcg'] == 0.5

@pytest.mark.parametrize('retriever_with_docs', ['tfidf'], indirect=True)
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
def test_faq_calculate_metrics(retriever_with_docs):
    if False:
        return 10
    pipeline = FAQPipeline(retriever=retriever_with_docs)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=EVAL_LABELS, params={'Retriever': {'top_k': 5}})
    metrics = eval_result.calculate_metrics(document_scope='document_id')
    assert 'Retriever' in eval_result
    assert 'Docs2Answers' in eval_result
    assert len(eval_result) == 2
    assert metrics['Retriever']['mrr'] == 0.5
    assert metrics['Retriever']['map'] == 0.5
    assert metrics['Retriever']['recall_multi_hit'] == 0.5
    assert metrics['Retriever']['recall_single_hit'] == 0.5
    assert metrics['Retriever']['precision'] == 0.1
    assert metrics['Retriever']['ndcg'] == 0.5
    assert metrics['Docs2Answers']['exact_match'] == 0.0
    assert metrics['Docs2Answers']['f1'] == 0.0

@pytest.mark.parametrize('retriever_with_docs', ['tfidf'], indirect=True)
@pytest.mark.parametrize('document_store_with_docs', ['memory'], indirect=True)
def test_question_generation_eval(retriever_with_docs, question_generator):
    if False:
        print('Hello World!')
    pipeline = RetrieverQuestionGenerationPipeline(retriever=retriever_with_docs, question_generator=question_generator)
    eval_result: EvaluationResult = pipeline.eval_batch(labels=EVAL_LABELS, params={'Retriever': {'top_k': 5}})
    metrics = eval_result.calculate_metrics(document_scope='document_id')
    assert 'Retriever' in eval_result
    assert 'QuestionGenerator' in eval_result
    assert len(eval_result) == 2
    assert metrics['Retriever']['mrr'] == 0.5
    assert metrics['Retriever']['map'] == 0.5
    assert metrics['Retriever']['recall_multi_hit'] == 0.5
    assert metrics['Retriever']['recall_single_hit'] == 0.5
    assert metrics['Retriever']['precision'] == 0.1
    assert metrics['Retriever']['ndcg'] == 0.5
    assert metrics['QuestionGenerator']['mrr'] == 0.5
    assert metrics['QuestionGenerator']['map'] == 0.5
    assert metrics['QuestionGenerator']['recall_multi_hit'] == 0.5
    assert metrics['QuestionGenerator']['recall_single_hit'] == 0.5
    assert metrics['QuestionGenerator']['precision'] == 0.1
    assert metrics['QuestionGenerator']['ndcg'] == 0.5