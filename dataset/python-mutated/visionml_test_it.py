import unittest
import pytest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
try:
    from apache_beam.ml.gcp.visionml import AnnotateImage
    from google.cloud import vision
except ImportError:
    vision = None

def extract(response):
    if False:
        return 10
    for r in response.responses:
        for text_annotation in r.text_annotations:
            yield text_annotation.description

@pytest.mark.it_postcommit
@unittest.skipIf(vision is None, 'GCP dependencies are not installed')
class VisionMlTestIT(unittest.TestCase):

    def test_text_detection_with_language_hint(self):
        if False:
            i = 10
            return i + 15
        IMAGES_TO_ANNOTATE = ['gs://apache-beam-samples/advanced_analytics/vision/sign.jpg']
        IMAGE_CONTEXT = [vision.ImageContext({'language_hints': ['en']})]
        with TestPipeline(is_integration_test=True) as p:
            contexts = p | 'Create context' >> beam.Create(dict(zip(IMAGES_TO_ANNOTATE, IMAGE_CONTEXT)))
            output = p | beam.Create(IMAGES_TO_ANNOTATE) | AnnotateImage(features=[vision.Feature({'type_': vision.Feature.Type.TEXT_DETECTION})], context_side_input=beam.pvalue.AsDict(contexts)) | beam.ParDo(extract)
        assert_that(output, equal_to(['WAITING?\nPLEASE\nTURN OFF\nYOUR\nENGINE', 'WAITING?', 'PLEASE', 'TURN', 'OFF', 'YOUR', 'ENGINE']))
if __name__ == '__main__':
    unittest.main()