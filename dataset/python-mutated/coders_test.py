"""Test for the coders example."""
import logging
import unittest
import apache_beam as beam
from apache_beam.examples.cookbook import coders
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

class CodersTest(unittest.TestCase):
    SAMPLE_RECORDS = [{'host': ['Germany', 1], 'guest': ['Italy', 0]}, {'host': ['Germany', 1], 'guest': ['Brasil', 3]}, {'host': ['Brasil', 1], 'guest': ['Italy', 0]}]
    EXPECTED_RESULT = [('Italy', 0), ('Brasil', 6), ('Germany', 3)]

    def test_compute_points(self):
        if False:
            return 10
        with TestPipeline() as p:
            records = p | 'create' >> beam.Create(self.SAMPLE_RECORDS)
            result = records | 'points' >> beam.FlatMap(coders.compute_points) | beam.CombinePerKey(sum)
            assert_that(result, equal_to(self.EXPECTED_RESULT))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()