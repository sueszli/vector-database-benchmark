"""Test for the BigQuery tornadoes example."""
import logging
import unittest
import apache_beam as beam
from apache_beam.examples.cookbook import bigquery_tornadoes
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

class BigQueryTornadoesTest(unittest.TestCase):

    def test_basics(self):
        if False:
            return 10
        with TestPipeline() as p:
            rows = p | 'create' >> beam.Create([{'month': 1, 'day': 1, 'tornado': False}, {'month': 1, 'day': 2, 'tornado': True}, {'month': 1, 'day': 3, 'tornado': True}, {'month': 2, 'day': 1, 'tornado': True}])
            results = bigquery_tornadoes.count_tornadoes(rows)
            assert_that(results, equal_to([{'month': 1, 'tornado_count': 2}, {'month': 2, 'tornado_count': 1}]))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()