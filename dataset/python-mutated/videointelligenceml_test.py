"""Unit tests for videointelligenceml."""
import logging
import unittest
import mock
import apache_beam as beam
from apache_beam.metrics import MetricsFilter
from apache_beam.typehints.decorators import TypeCheckError
try:
    from google.cloud.videointelligence import VideoIntelligenceServiceClient
    from google.cloud import videointelligence
    from apache_beam.ml.gcp import videointelligenceml
except ImportError:
    VideoIntelligenceServiceClient = None

@unittest.skipIf(VideoIntelligenceServiceClient is None, 'Video intelligence dependencies are not installed')
class VideoIntelligenceTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._mock_client = mock.Mock()
        self.m2 = mock.Mock()
        self.m2.result.return_value = None
        self._mock_client.annotate_video.return_value = self.m2
        self.features = [videointelligence.Feature.LABEL_DETECTION]
        self.location_id = 'us-west1'
        config = videointelligence.SpeechTranscriptionConfig(language_code='en-US', enable_automatic_punctuation=True)
        self.video_ctx = videointelligence.VideoContext(speech_transcription_config=config)

    def test_AnnotateVideo_with_side_input_context(self):
        if False:
            while True:
                i = 10
        videos_to_annotate = ['gs://cloud-samples-data/video/cat.mp4', 'gs://some-other-video/sample.mp4', 'gs://some-other-video/sample_2.mp4']
        video_contexts = [('gs://cloud-samples-data/video/cat.mp4', self.video_ctx), ('gs://some-other-video/sample.mp4', self.video_ctx)]
        expected_counter = len(videos_to_annotate)
        with mock.patch.object(videointelligenceml, 'get_videointelligence_client', return_value=self._mock_client):
            p = beam.Pipeline()
            context_side_input = p | 'Video contexts' >> beam.Create(video_contexts)
            _ = p | 'Create data' >> beam.Create(videos_to_annotate) | 'Annotate video' >> videointelligenceml.AnnotateVideo(self.features, context_side_input=beam.pvalue.AsDict(context_side_input))
            result = p.run()
            result.wait_until_finish()
            read_filter = MetricsFilter().with_name('API Calls')
            query_result = result.metrics().query(read_filter)
            if query_result['counters']:
                read_counter = query_result['counters'][0]
                self.assertTrue(read_counter.committed == expected_counter)

    def test_AnnotateVideo_URIs(self):
        if False:
            while True:
                i = 10
        videos_to_annotate = ['gs://cloud-samples-data/video/cat.mp4', 'gs://cloud-samples-data/video/cat.mp4']
        expected_counter = len(videos_to_annotate)
        with mock.patch.object(videointelligenceml, 'get_videointelligence_client', return_value=self._mock_client):
            p = beam.Pipeline()
            _ = p | 'Create data' >> beam.Create(videos_to_annotate) | 'Annotate video' >> videointelligenceml.AnnotateVideo(self.features)
            result = p.run()
            result.wait_until_finish()
            read_filter = MetricsFilter().with_name('API Calls')
            query_result = result.metrics().query(read_filter)
            if query_result['counters']:
                read_counter = query_result['counters'][0]
                self.assertTrue(read_counter.committed == expected_counter)

    def test_AnnotateVideoWithContext_b64_content(self):
        if False:
            while True:
                i = 10
        base_64_encoded_video = b'YmVnaW4gNjQ0IGNhdC12aWRlby5tcDRNICAgICgmOVQ+NyFNPCMwUi4uZmFrZV92aWRlb'
        videos_to_annotate = [(base_64_encoded_video, self.video_ctx), (base_64_encoded_video, None), (base_64_encoded_video, self.video_ctx)]
        expected_counter = len(videos_to_annotate)
        with mock.patch.object(videointelligenceml, 'get_videointelligence_client', return_value=self._mock_client):
            p = beam.Pipeline()
            _ = p | 'Create data' >> beam.Create(videos_to_annotate) | 'Annotate video' >> videointelligenceml.AnnotateVideoWithContext(self.features)
            result = p.run()
            result.wait_until_finish()
        read_filter = MetricsFilter().with_name('API Calls')
        query_result = result.metrics().query(read_filter)
        if query_result['counters']:
            read_counter = query_result['counters'][0]
            self.assertTrue(read_counter.committed == expected_counter)

    def test_AnnotateVideo_b64_content(self):
        if False:
            for i in range(10):
                print('nop')
        base_64_encoded_video = b'YmVnaW4gNjQ0IGNhdC12aWRlby5tcDRNICAgICgmOVQ+NyFNPCMwUi4uZmFrZV92aWRlb'
        videos_to_annotate = [base_64_encoded_video, base_64_encoded_video, base_64_encoded_video]
        expected_counter = len(videos_to_annotate)
        with mock.patch.object(videointelligenceml, 'get_videointelligence_client', return_value=self._mock_client):
            p = beam.Pipeline()
            _ = p | 'Create data' >> beam.Create(videos_to_annotate) | 'Annotate video' >> videointelligenceml.AnnotateVideo(self.features)
            result = p.run()
            result.wait_until_finish()
            read_filter = MetricsFilter().with_name('API Calls')
            query_result = result.metrics().query(read_filter)
            if query_result['counters']:
                read_counter = query_result['counters'][0]
                self.assertTrue(read_counter.committed == expected_counter)

    def test_AnnotateVideoWithContext_bad_input(self):
        if False:
            i = 10
            return i + 15
        'AnnotateVideoWithContext should not accept videos without context'
        videos_to_annotate = ['gs://cloud-samples-data/video/cat.mp4', 'gs://cloud-samples-data/video/cat.mp4']
        with mock.patch.object(videointelligenceml, 'get_videointelligence_client', return_value=self._mock_client):
            with self.assertRaises(TypeCheckError):
                p = beam.Pipeline()
                _ = p | 'Create data' >> beam.Create(videos_to_annotate) | 'Annotate video' >> videointelligenceml.AnnotateVideoWithContext(self.features)
                result = p.run()
                result.wait_until_finish()

    def test_AnnotateVideo_bad_input(self):
        if False:
            i = 10
            return i + 15
        videos_to_annotate = [123456789, 123456789, 123456789]
        with mock.patch.object(videointelligenceml, 'get_videointelligence_client', return_value=self._mock_client):
            with self.assertRaises(TypeCheckError):
                p = beam.Pipeline()
                _ = p | 'Create data' >> beam.Create(videos_to_annotate) | 'Annotate video' >> videointelligenceml.AnnotateVideo(self.features)
                result = p.run()
                result.wait_until_finish()
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()