from __future__ import annotations
from unittest.mock import patch
from google.cloud.language_v1 import AnalyzeEntitiesResponse, AnalyzeEntitySentimentResponse, AnalyzeSentimentResponse, ClassifyTextResponse, Document
from airflow.providers.google.cloud.operators.natural_language import CloudNaturalLanguageAnalyzeEntitiesOperator, CloudNaturalLanguageAnalyzeEntitySentimentOperator, CloudNaturalLanguageAnalyzeSentimentOperator, CloudNaturalLanguageClassifyTextOperator
DOCUMENT = Document(content='Airflow is a platform to programmatically author, schedule and monitor workflows.')
CLASSIFY_TEXT_RESPONSE = ClassifyTextResponse()
ANALYZE_ENTITIES_RESPONSE = AnalyzeEntitiesResponse()
ANALYZE_ENTITY_SENTIMENT_RESPONSE = AnalyzeEntitySentimentResponse()
ANALYZE_SENTIMENT_RESPONSE = AnalyzeSentimentResponse()
ENCODING_TYPE = 'UTF32'

class TestCloudLanguageAnalyzeEntitiesOperator:

    @patch('airflow.providers.google.cloud.operators.natural_language.CloudNaturalLanguageHook')
    def test_minimal_green_path(self, hook_mock):
        if False:
            return 10
        hook_mock.return_value.analyze_entities.return_value = ANALYZE_ENTITIES_RESPONSE
        op = CloudNaturalLanguageAnalyzeEntitiesOperator(task_id='task-id', document=DOCUMENT)
        resp = op.execute({})
        assert resp == {}

class TestCloudLanguageAnalyzeEntitySentimentOperator:

    @patch('airflow.providers.google.cloud.operators.natural_language.CloudNaturalLanguageHook')
    def test_minimal_green_path(self, hook_mock):
        if False:
            return 10
        hook_mock.return_value.analyze_entity_sentiment.return_value = ANALYZE_ENTITY_SENTIMENT_RESPONSE
        op = CloudNaturalLanguageAnalyzeEntitySentimentOperator(task_id='task-id', document=DOCUMENT)
        resp = op.execute({})
        assert resp == {}

class TestCloudLanguageAnalyzeSentimentOperator:

    @patch('airflow.providers.google.cloud.operators.natural_language.CloudNaturalLanguageHook')
    def test_minimal_green_path(self, hook_mock):
        if False:
            return 10
        hook_mock.return_value.analyze_sentiment.return_value = ANALYZE_SENTIMENT_RESPONSE
        op = CloudNaturalLanguageAnalyzeSentimentOperator(task_id='task-id', document=DOCUMENT)
        resp = op.execute({})
        assert resp == {}

class TestCloudLanguageClassifyTextOperator:

    @patch('airflow.providers.google.cloud.operators.natural_language.CloudNaturalLanguageHook')
    def test_minimal_green_path(self, hook_mock):
        if False:
            for i in range(10):
                print('nop')
        hook_mock.return_value.classify_text.return_value = CLASSIFY_TEXT_RESPONSE
        op = CloudNaturalLanguageClassifyTextOperator(task_id='task-id', document=DOCUMENT)
        resp = op.execute({})
        assert resp == {}