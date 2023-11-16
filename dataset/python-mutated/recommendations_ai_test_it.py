"""Integration tests for Recommendations AI transforms."""
from __future__ import absolute_import
import random
import unittest
from datetime import datetime
import pytest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.testing.util import is_not_empty
try:
    from google.cloud import recommendationengine
    from apache_beam.ml.gcp import recommendations_ai
except ImportError:
    recommendationengine = None
GCP_TEST_PROJECT = 'apache-beam-testing'
CATALOG_ITEM = {'id': f'aitest-{int(datetime.now().timestamp())}-{int(random.randint(1, 10000))}', 'title': 'Sample laptop', 'description': 'Indisputably the most fantastic laptop ever created.', 'language_code': 'en', 'category_hierarchies': [{'categories': ['Electronic', 'Computers']}]}

def extract_id(response):
    if False:
        i = 10
        return i + 15
    yield response['id']

def extract_event_type(response):
    if False:
        i = 10
        return i + 15
    yield response['event_type']

def extract_prediction(response):
    if False:
        while True:
            i = 10
    yield response[0]['results']

@pytest.mark.it_postcommit
@unittest.skipIf(recommendationengine is None, 'Recommendations AI dependencies not installed.')
class RecommendationAIIT(unittest.TestCase):
    test_ran = False

    def test_create_catalog_item(self):
        if False:
            print('Hello World!')
        with TestPipeline(is_integration_test=True) as p:
            RecommendationAIIT.test_ran = True
            output = p | 'Create data' >> beam.Create([CATALOG_ITEM]) | 'Create CatalogItem' >> recommendations_ai.CreateCatalogItem(project=GCP_TEST_PROJECT) | beam.ParDo(extract_id) | beam.combiners.ToList()
            assert_that(output, equal_to([[CATALOG_ITEM['id']]]))

    def test_create_user_event(self):
        if False:
            while True:
                i = 10
        USER_EVENT = {'event_type': 'page-visit', 'user_info': {'visitor_id': '1'}}
        with TestPipeline(is_integration_test=True) as p:
            RecommendationAIIT.test_ran = True
            output = p | 'Create data' >> beam.Create([USER_EVENT]) | 'Create UserEvent' >> recommendations_ai.WriteUserEvent(project=GCP_TEST_PROJECT) | beam.ParDo(extract_event_type) | beam.combiners.ToList()
            assert_that(output, equal_to([[USER_EVENT['event_type']]]))

    def test_predict(self):
        if False:
            print('Hello World!')
        USER_EVENT = {'event_type': 'page-visit', 'user_info': {'visitor_id': '1'}}
        with TestPipeline(is_integration_test=True) as p:
            RecommendationAIIT.test_ran = True
            output = p | 'Create data' >> beam.Create([USER_EVENT]) | 'Predict UserEvent' >> recommendations_ai.PredictUserEvent(project=GCP_TEST_PROJECT, placement_id='recently_viewed_default') | beam.ParDo(extract_prediction)
            assert_that(output, is_not_empty())

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        if not cls.test_ran:
            raise unittest.SkipTest('all test skipped')
        client = recommendationengine.CatalogServiceClient()
        parent = f'projects/{GCP_TEST_PROJECT}/locations/global/catalogs/default_catalog'
        for item in list(client.list_catalog_items(parent=parent)):
            client.delete_catalog_item(name=f'projects/{GCP_TEST_PROJECT}/locations/global/catalogs/default_catalog/catalogItems/{item.id}')
if __name__ == '__main__':
    unittest.main()