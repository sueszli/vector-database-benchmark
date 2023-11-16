"""Test for the TF-IDF example."""
import logging
import unittest
import apache_beam as beam
from apache_beam.examples.complete import tfidf
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
EXPECTED_RESULTS = set([('ghi', '1.txt', 0.3662040962227032), ('abc', '1.txt', 0.0), ('abc', '3.txt', 0.0), ('abc', '2.txt', 0.0), ('def', '1.txt', 0.13515503603605478), ('def', '2.txt', 0.2027325540540822)])

class TfIdfTest(unittest.TestCase):

    def test_tfidf_transform(self):
        if False:
            i = 10
            return i + 15
        with TestPipeline() as p:

            def re_key(word_uri_tfidf):
                if False:
                    while True:
                        i = 10
                (word, (uri, tfidf)) = word_uri_tfidf
                return (word, uri, tfidf)
            uri_to_line = p | 'create sample' >> beam.Create([('1.txt', 'abc def ghi'), ('2.txt', 'abc def'), ('3.txt', 'abc')])
            result = uri_to_line | tfidf.TfIdf() | beam.Map(re_key)
            assert_that(result, equal_to(EXPECTED_RESULTS))
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()