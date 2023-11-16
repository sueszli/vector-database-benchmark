"""Tests for apache_beam.runners.interactive.options.capture_control."""
import unittest
from unittest.mock import patch
import apache_beam as beam
from apache_beam import coders
from apache_beam.portability.api import beam_interactive_api_pb2
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.runners import runner
from apache_beam.runners.interactive import background_caching_job as bcj
from apache_beam.runners.interactive import interactive_beam as ib
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive import interactive_runner
from apache_beam.runners.interactive.caching.streaming_cache import StreamingCache
from apache_beam.runners.interactive.options import capture_control
from apache_beam.runners.interactive.options import capture_limiters
from apache_beam.testing.test_stream_service import TestStreamServiceController

def _build_an_empty_streaming_pipeline():
    if False:
        print('Hello World!')
    from apache_beam.options.pipeline_options import PipelineOptions
    from apache_beam.options.pipeline_options import StandardOptions
    pipeline_options = PipelineOptions()
    pipeline_options.view_as(StandardOptions).streaming = True
    p = beam.Pipeline(interactive_runner.InteractiveRunner(), options=pipeline_options)
    ib.watch({'pipeline': p})
    return p

def _fake_a_running_test_stream_service(pipeline):
    if False:
        i = 10
        return i + 15

    class FakeReader:

        def read_multiple(self):
            if False:
                while True:
                    i = 10
            yield 1
    test_stream_service = TestStreamServiceController(FakeReader())
    test_stream_service.start()
    ie.current_env().set_test_stream_service_controller(pipeline, test_stream_service)

@unittest.skipIf(not ie.current_env().is_interactive_ready, '[interactive] dependency is not installed.')
class CaptureControlTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        ie.new_env()

    @patch('apache_beam.runners.interactive.background_caching_job.BackgroundCachingJob.cancel')
    @patch('apache_beam.testing.test_stream_service.TestStreamServiceController.stop')
    def test_capture_control_evict_captured_data(self, mocked_test_stream_service_stop, mocked_background_caching_job_cancel):
        if False:
            return 10
        p = _build_an_empty_streaming_pipeline()
        ie.current_env().track_user_pipelines()
        self.assertFalse(ie.current_env().tracked_user_pipelines == set())
        background_caching_job = bcj.BackgroundCachingJob(runner.PipelineResult(runner.PipelineState.RUNNING), limiters=[])
        ie.current_env().set_background_caching_job(p, background_caching_job)
        _fake_a_running_test_stream_service(p)
        background_caching_job._pipeline_result = runner.PipelineResult(runner.PipelineState.CANCELLING)
        self.assertIsNotNone(ie.current_env().get_test_stream_service_controller(p))
        ie.current_env().set_cached_source_signature(p, 'a signature')
        ie.current_env().mark_pcollection_computed(['fake_pcoll'])
        capture_control.evict_captured_data()
        mocked_background_caching_job_cancel.assert_called()
        mocked_test_stream_service_stop.assert_called_once()
        self.assertFalse(background_caching_job.is_done())
        self.assertIsNone(ie.current_env().get_test_stream_service_controller(p))
        self.assertTrue(ie.current_env().computed_pcollections == set())
        self.assertTrue(ie.current_env().get_cached_source_signature(p) == set())

    def test_capture_size_limit_not_reached_when_no_cache(self):
        if False:
            while True:
                i = 10
        self.assertEqual(len(ie.current_env()._cache_managers), 0)
        limiter = capture_limiters.SizeLimiter(1)
        self.assertFalse(limiter.is_triggered())

    def test_capture_size_limit_not_reached_when_no_file(self):
        if False:
            while True:
                i = 10
        cache = StreamingCache(cache_dir=None)
        self.assertFalse(cache.exists('my_label'))
        ie.current_env().set_cache_manager(cache, 'dummy pipeline')
        limiter = capture_limiters.SizeLimiter(1)
        self.assertFalse(limiter.is_triggered())

    def test_capture_size_limit_not_reached_when_file_size_under_limit(self):
        if False:
            for i in range(10):
                print('nop')
        ib.options.capture_size_limit = 100
        cache = StreamingCache(cache_dir=None)
        cache.sink(['my_label'], is_capture=True)
        cache.write([beam_interactive_api_pb2.TestStreamFileRecord()], 'my_label')
        self.assertTrue(cache.exists('my_label'))
        ie.current_env().set_cache_manager(cache, 'dummy pipeline')
        limiter = capture_limiters.SizeLimiter(ib.options.capture_size_limit)
        self.assertFalse(limiter.is_triggered())

    def test_capture_size_limit_reached_when_file_size_above_limit(self):
        if False:
            while True:
                i = 10
        ib.options.capture_size_limit = 1
        cache = StreamingCache(cache_dir=None)
        cache.sink(['my_label'], is_capture=True)
        cache.write([beam_interactive_api_pb2.TestStreamFileRecord(recorded_event=beam_runner_api_pb2.TestStreamPayload.Event(element_event=beam_runner_api_pb2.TestStreamPayload.Event.AddElements(elements=[beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coders.FastPrimitivesCoder().encode('a'), timestamp=0)])))], 'my_label')
        self.assertTrue(cache.exists('my_label'))
        p = _build_an_empty_streaming_pipeline()
        ie.current_env().set_cache_manager(cache, p)
        limiter = capture_limiters.SizeLimiter(1)
        self.assertTrue(limiter.is_triggered())

    def test_timer_terminates_capture_size_checker(self):
        if False:
            i = 10
            return i + 15
        p = _build_an_empty_streaming_pipeline()

        class FakeLimiter(capture_limiters.Limiter):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.trigger = False

            def is_triggered(self):
                if False:
                    print('Hello World!')
                return self.trigger
        limiter = FakeLimiter()
        background_caching_job = bcj.BackgroundCachingJob(runner.PipelineResult(runner.PipelineState.CANCELLING), limiters=[limiter])
        ie.current_env().set_background_caching_job(p, background_caching_job)
        self.assertFalse(background_caching_job.is_done())
        limiter.trigger = True
        self.assertTrue(background_caching_job.is_done())
if __name__ == '__main__':
    unittest.main()