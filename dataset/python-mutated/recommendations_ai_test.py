"""Unit tests for Recommendations AI transforms."""
from __future__ import absolute_import
import unittest
import mock
import apache_beam as beam
from apache_beam.metrics import MetricsFilter
try:
    from google.cloud import recommendationengine
    from apache_beam.ml.gcp import recommendations_ai
except ImportError:
    recommendationengine = None

@unittest.skipIf(recommendationengine is None, 'Recommendations AI dependencies not installed.')
class RecommendationsAICatalogItemTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._mock_client = mock.Mock()
        self._mock_client.create_catalog_item.return_value = recommendationengine.CatalogItem()
        self.m2 = mock.Mock()
        self.m2.result.return_value = None
        self._mock_client.import_catalog_items.return_value = self.m2
        self._catalog_item = {'id': '12345', 'title': 'Sample laptop', 'description': 'Indisputably the most fantastic laptop ever created.', 'language_code': 'en', 'category_hierarchies': [{'categories': ['Electronic', 'Computers']}]}

    def test_CreateCatalogItem(self):
        if False:
            for i in range(10):
                print('nop')
        expected_counter = 1
        with mock.patch.object(recommendations_ai, 'get_recommendation_catalog_client', return_value=self._mock_client):
            p = beam.Pipeline()
            _ = p | 'Create data' >> beam.Create([self._catalog_item]) | 'Create CatalogItem' >> recommendations_ai.CreateCatalogItem(project='test')
            result = p.run()
            result.wait_until_finish()
            read_filter = MetricsFilter().with_name('api_calls')
            query_result = result.metrics().query(read_filter)
            if query_result['counters']:
                read_counter = query_result['counters'][0]
                self.assertTrue(read_counter.result == expected_counter)

    def test_ImportCatalogItems(self):
        if False:
            print('Hello World!')
        expected_counter = 1
        with mock.patch.object(recommendations_ai, 'get_recommendation_catalog_client', return_value=self._mock_client):
            p = beam.Pipeline()
            _ = p | 'Create data' >> beam.Create([(self._catalog_item['id'], self._catalog_item), (self._catalog_item['id'], self._catalog_item)]) | 'Create CatalogItems' >> recommendations_ai.ImportCatalogItems(project='test')
            result = p.run()
            result.wait_until_finish()
            read_filter = MetricsFilter().with_name('api_calls')
            query_result = result.metrics().query(read_filter)
            if query_result['counters']:
                read_counter = query_result['counters'][0]
                self.assertTrue(read_counter.result == expected_counter)

@unittest.skipIf(recommendationengine is None, 'Recommendations AI dependencies not installed.')
class RecommendationsAIUserEventTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self._mock_client = mock.Mock()
        self._mock_client.write_user_event.return_value = recommendationengine.UserEvent()
        self.m2 = mock.Mock()
        self.m2.result.return_value = None
        self._mock_client.import_user_events.return_value = self.m2
        self._user_event = {'event_type': 'page-visit', 'user_info': {'visitor_id': '1'}}

    def test_CreateUserEvent(self):
        if False:
            print('Hello World!')
        expected_counter = 1
        with mock.patch.object(recommendations_ai, 'get_recommendation_user_event_client', return_value=self._mock_client):
            p = beam.Pipeline()
            _ = p | 'Create data' >> beam.Create([self._user_event]) | 'Create UserEvent' >> recommendations_ai.WriteUserEvent(project='test')
            result = p.run()
            result.wait_until_finish()
            read_filter = MetricsFilter().with_name('api_calls')
            query_result = result.metrics().query(read_filter)
            if query_result['counters']:
                read_counter = query_result['counters'][0]
                self.assertTrue(read_counter.result == expected_counter)

    def test_ImportUserEvents(self):
        if False:
            for i in range(10):
                print('nop')
        expected_counter = 1
        with mock.patch.object(recommendations_ai, 'get_recommendation_user_event_client', return_value=self._mock_client):
            p = beam.Pipeline()
            _ = p | 'Create data' >> beam.Create([(self._user_event['user_info']['visitor_id'], self._user_event), (self._user_event['user_info']['visitor_id'], self._user_event)]) | 'Create UserEvents' >> recommendations_ai.ImportUserEvents(project='test')
            result = p.run()
            result.wait_until_finish()
            read_filter = MetricsFilter().with_name('api_calls')
            query_result = result.metrics().query(read_filter)
            if query_result['counters']:
                read_counter = query_result['counters'][0]
                self.assertTrue(read_counter.result == expected_counter)

@unittest.skipIf(recommendationengine is None, 'Recommendations AI dependencies not installed.')
class RecommendationsAIPredictTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._mock_client = mock.Mock()
        self._mock_client.predict.return_value = [recommendationengine.PredictResponse()]
        self._user_event = {'event_type': 'page-visit', 'user_info': {'visitor_id': '1'}}

    def test_Predict(self):
        if False:
            return 10
        expected_counter = 1
        with mock.patch.object(recommendations_ai, 'get_recommendation_prediction_client', return_value=self._mock_client):
            p = beam.Pipeline()
            _ = p | 'Create data' >> beam.Create([self._user_event]) | 'Prediction UserEvents' >> recommendations_ai.PredictUserEvent(project='test', placement_id='recently_viewed_default')
            result = p.run()
            result.wait_until_finish()
            read_filter = MetricsFilter().with_name('api_calls')
            query_result = result.metrics().query(read_filter)
            if query_result['counters']:
                read_counter = query_result['counters'][0]
                self.assertTrue(read_counter.result == expected_counter)
if __name__ == '__main__':
    unittest.main()