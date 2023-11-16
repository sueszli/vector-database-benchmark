"""Unit test for Bigquery verifier"""
import logging
import unittest
import mock
from hamcrest import assert_that as hc_assert_that
from apache_beam.io.gcp import bigquery_tools
from apache_beam.io.gcp.tests import bigquery_matcher as bq_verifier
from apache_beam.testing.test_utils import patch_retry
try:
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
except ImportError:
    bigquery = None
    NotFound = None

@unittest.skipIf(bigquery is None, 'Bigquery dependencies are not installed.')
@mock.patch.object(bigquery, 'Client')
class BigqueryMatcherTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._mock_result = mock.Mock()
        patch_retry(self, bq_verifier)

    def test_bigquery_matcher_success(self, mock_bigquery):
        if False:
            print('Hello World!')
        mock_query_result = [mock.Mock(), mock.Mock(), mock.Mock()]
        mock_query_result[0].values.return_value = []
        mock_query_result[1].values.return_value = None
        mock_query_result[2].values.return_value = None
        mock_query = mock_bigquery.return_value.query
        mock_query.return_value.result.return_value = mock_query_result
        matcher = bq_verifier.BigqueryMatcher('mock_project', 'mock_query', '59f9d6bdee30d67ea73b8aded121c3a0280f9cd8')
        hc_assert_that(self._mock_result, matcher)
        self.assertEqual(1, mock_query.call_count)

    def test_bigquery_matcher_success_streaming_retry(self, mock_bigquery):
        if False:
            for i in range(10):
                print('nop')
        empty_query_result = []
        mock_query_result = [mock.Mock(), mock.Mock(), mock.Mock()]
        mock_query_result[0].values.return_value = []
        mock_query_result[1].values.return_value = None
        mock_query_result[2].values.return_value = None
        mock_query = mock_bigquery.return_value.query
        mock_query.return_value.result.side_effect = [empty_query_result, mock_query_result]
        matcher = bq_verifier.BigqueryMatcher('mock_project', 'mock_query', '59f9d6bdee30d67ea73b8aded121c3a0280f9cd8', timeout_secs=5)
        hc_assert_that(self._mock_result, matcher)
        self.assertEqual(2, mock_query.call_count)

    def test_bigquery_matcher_query_error_retry(self, mock_bigquery):
        if False:
            return 10
        mock_query = mock_bigquery.return_value.query
        mock_query.side_effect = NotFound('table not found')
        matcher = bq_verifier.BigqueryMatcher('mock_project', 'mock_query', 'mock_checksum')
        with self.assertRaises(NotFound):
            hc_assert_that(self._mock_result, matcher)
        self.assertEqual(bq_verifier.MAX_RETRIES + 1, mock_query.call_count)

    def test_bigquery_matcher_query_error_checksum(self, mock_bigquery):
        if False:
            return 10
        empty_query_result = []
        mock_query = mock_bigquery.return_value.query
        mock_query.return_value.result.return_value = empty_query_result
        matcher = bq_verifier.BigqueryMatcher('mock_project', 'mock_query', '59f9d6bdee30d67ea73b8aded121c3a0280f9cd8')
        with self.assertRaisesRegex(AssertionError, 'Expected checksum'):
            hc_assert_that(self._mock_result, matcher)
        self.assertEqual(1, mock_query.call_count)

@unittest.skipIf(bigquery is None, 'Bigquery dependencies are not installed.')
@mock.patch.object(bigquery_tools, 'BigQueryWrapper')
class BigqueryTableMatcherTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._mock_result = mock.Mock()
        patch_retry(self, bq_verifier)

    def test_bigquery_table_matcher_success(self, mock_bigquery):
        if False:
            return 10
        mock_query_result = mock.Mock(partitioning='a lot of partitioning', clustering={'column': 'FRIENDS'})
        mock_bigquery.return_value.get_table.return_value = mock_query_result
        matcher = bq_verifier.BigQueryTableMatcher('mock_project', 'mock_dataset', 'mock_table', {'partitioning': 'a lot of partitioning', 'clustering': {'column': 'FRIENDS'}})
        hc_assert_that(self._mock_result, matcher)

    def test_bigquery_table_matcher_query_error_retry(self, mock_bigquery):
        if False:
            return 10
        mock_query = mock_bigquery.return_value.get_table
        mock_query.side_effect = ValueError('table not found')
        matcher = bq_verifier.BigQueryTableMatcher('mock_project', 'mock_dataset', 'mock_table', {'partitioning': 'a lot of partitioning', 'clustering': {'column': 'FRIENDS'}})
        with self.assertRaises(ValueError):
            hc_assert_that(self._mock_result, matcher)
        self.assertEqual(bq_verifier.MAX_RETRIES + 1, mock_query.call_count)

@unittest.skipIf(bigquery is None, 'Bigquery dependencies are not installed.')
@mock.patch.object(bigquery, 'Client')
class BigqueryFullResultStreamingMatcherTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.timeout = 0.01

    def test__get_query_result_timeout(self, mock_bigquery):
        if False:
            print('Hello World!')
        mock_query = mock_bigquery.return_value.query
        mock_query.return_value.result.return_value = []
        matcher = bq_verifier.BigqueryFullResultStreamingMatcher('some-project', 'some-query', [1, 2, 3], timeout=self.timeout)
        with self.assertRaises(TimeoutError):
            matcher._get_query_result()
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()