"""Test for the autocomplete example."""
import unittest
import pytest
import apache_beam as beam
from apache_beam.examples.complete import autocomplete
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.test_utils import compute_hash
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

class AutocompleteTest(unittest.TestCase):
    WORDS = ['this', 'this', 'that', 'to', 'to', 'to']
    KINGLEAR_HASH_SUM = 268011785062540
    KINGLEAR_INPUT = 'gs://dataflow-samples/shakespeare/kinglear.txt'
    EXPECTED_PREFIXES = [('t', ((3, 'to'), (2, 'this'), (1, 'that'))), ('to', ((3, 'to'),)), ('th', ((2, 'this'), (1, 'that'))), ('thi', ((2, 'this'),)), ('this', ((2, 'this'),)), ('tha', ((1, 'that'),)), ('that', ((1, 'that'),))]

    def test_top_prefixes(self):
        if False:
            print('Hello World!')
        with TestPipeline() as p:
            words = p | beam.Create(self.WORDS)
            result = words | autocomplete.TopPerPrefix(5)
            result = result | beam.Map(lambda k_vs: (k_vs[0], tuple(k_vs[1])))
            assert_that(result, equal_to(self.EXPECTED_PREFIXES))

    @pytest.mark.it_postcommit
    def test_autocomplete_it(self):
        if False:
            while True:
                i = 10
        with TestPipeline(is_integration_test=True) as p:
            words = p | beam.io.ReadFromText(self.KINGLEAR_INPUT)
            result = words | autocomplete.TopPerPrefix(10)
            result = result | beam.Map(lambda k_vs: [k_vs[0], k_vs[1][0][0], k_vs[1][0][1]])
            checksum = result | beam.Map(lambda x: int(compute_hash(x)[:8], 16)) | beam.CombineGlobally(sum)
            assert_that(checksum, equal_to([self.KINGLEAR_HASH_SUM]))
if __name__ == '__main__':
    unittest.main()