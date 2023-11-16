import unittest
import pytest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
try:
    from apache_beam.ml.gcp.naturallanguageml import AnnotateText
    from apache_beam.ml.gcp.naturallanguageml import Document
    from apache_beam.ml.gcp.naturallanguageml import enums
    from apache_beam.ml.gcp.naturallanguageml import types
except ImportError:
    AnnotateText = None

def extract(response):
    if False:
        print('Hello World!')
    yield beam.pvalue.TaggedOutput('language', response.language)
    yield beam.pvalue.TaggedOutput('parts_of_speech', [enums.PartOfSpeech.Tag(x.part_of_speech.tag).name for x in response.tokens])

@pytest.mark.it_postcommit
@unittest.skipIf(AnnotateText is None, 'GCP dependencies are not installed')
class NaturalLanguageMlTestIT(unittest.TestCase):

    def test_analyzing_syntax(self):
        if False:
            i = 10
            return i + 15
        with TestPipeline(is_integration_test=True) as p:
            output = p | beam.Create([Document('Unified programming model.')]) | AnnotateText(types.AnnotateTextRequest.Features(extract_syntax=True)) | beam.ParDo(extract).with_outputs('language', 'parts_of_speech')
            assert_that(output.language, equal_to(['en']), label='verify_language')
            assert_that(output.parts_of_speech, equal_to([['ADJ', 'NOUN', 'NOUN', 'PUNCT']]), label='verify_parts_of_speech')
if __name__ == '__main__':
    unittest.main()