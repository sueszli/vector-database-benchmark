"""Unit tests for Google Cloud Natural Language API transform."""
import unittest
from apache_beam.metrics import MetricsFilter
try:
    from google.cloud import language
except ImportError:
    language = None
else:
    from apache_beam.ml.gcp import naturallanguageml

@unittest.skipIf(language is None, 'GCP dependencies are not installed')
class NaturalLanguageMlTest(unittest.TestCase):

    def assertCounterEqual(self, pipeline_result, counter_name, expected):
        if False:
            while True:
                i = 10
        metrics = pipeline_result.metrics().query(MetricsFilter().with_name(counter_name))
        try:
            counter = metrics['counters'][0]
            self.assertEqual(expected, counter.result)
        except IndexError:
            raise AssertionError('Counter "{}" was not found'.format(counter_name))

    def test_document_source(self):
        if False:
            while True:
                i = 10
        document = naturallanguageml.Document('Hello, world!')
        dict_ = naturallanguageml.Document.to_dict(document)
        self.assertTrue('content' in dict_)
        self.assertFalse('gcs_content_uri' in dict_)
        document = naturallanguageml.Document('gs://sample/location', from_gcs=True)
        dict_ = naturallanguageml.Document.to_dict(document)
        self.assertFalse('content' in dict_)
        self.assertTrue('gcs_content_uri' in dict_)
if __name__ == '__main__':
    unittest.main()