"""An integration test that labels entities appearing in a video and checks
if some expected entities were properly recognized."""
import unittest
import hamcrest as hc
import pytest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import matches_all
try:
    from apache_beam.ml.gcp.videointelligenceml import AnnotateVideoWithContext
    from google.cloud.videointelligence import enums
    from google.cloud.videointelligence import types
except ImportError:
    AnnotateVideoWithContext = None

def extract_entities_descriptions(response):
    if False:
        return 10
    for result in response.annotation_results:
        for segment in result.segment_presence_label_annotations:
            yield segment.entity.description

@pytest.mark.it_postcommit
@unittest.skipIf(AnnotateVideoWithContext is None, 'GCP dependencies are not installed')
class VideoIntelligenceMlTestIT(unittest.TestCase):
    VIDEO_PATH = 'gs://apache-beam-samples/advanced_analytics/video/gbikes_dinosaur.mp4'

    def test_label_detection_with_video_context(self):
        if False:
            print('Hello World!')
        with TestPipeline(is_integration_test=True) as p:
            output = p | beam.Create([(self.VIDEO_PATH, types.VideoContext(label_detection_config=types.LabelDetectionConfig(label_detection_mode=enums.LabelDetectionMode.SHOT_MODE, model='builtin/latest')))]) | AnnotateVideoWithContext(features=[enums.Feature.LABEL_DETECTION]) | beam.ParDo(extract_entities_descriptions) | beam.combiners.ToList()
            assert_that(output, matches_all([hc.has_item(hc.contains_string('bicycle'))]))
if __name__ == '__main__':
    unittest.main()