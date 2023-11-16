import time
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
import apache_beam as beam
from apache_beam import coders
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.portability.api import beam_interactive_api_pb2
from apache_beam.runners.interactive import background_caching_job as bcj
from apache_beam.runners.interactive import interactive_beam as ib
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive.caching.cacheable import CacheKey
from apache_beam.runners.interactive.interactive_runner import InteractiveRunner
from apache_beam.runners.interactive.options.capture_limiters import Limiter
from apache_beam.runners.interactive.recording_manager import ElementStream
from apache_beam.runners.interactive.recording_manager import Recording
from apache_beam.runners.interactive.recording_manager import RecordingManager
from apache_beam.runners.interactive.testing.test_cache_manager import FileRecordsBuilder
from apache_beam.runners.interactive.testing.test_cache_manager import InMemoryCache
from apache_beam.runners.runner import PipelineState
from apache_beam.testing.test_stream import TestStream
from apache_beam.testing.test_stream import WindowedValueHolder
from apache_beam.transforms.window import GlobalWindow
from apache_beam.utils.timestamp import MIN_TIMESTAMP
from apache_beam.utils.windowed_value import WindowedValue

class MockPipelineResult(beam.runners.runner.PipelineResult):
    """Mock class for controlling a PipelineResult."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self._state = PipelineState.RUNNING

    def wait_until_finish(self):
        if False:
            print('Hello World!')
        pass

    def set_state(self, state):
        if False:
            for i in range(10):
                print('nop')
        self._state = state

    @property
    def state(self):
        if False:
            print('Hello World!')
        return self._state

    def cancel(self):
        if False:
            while True:
                i = 10
        self._state = PipelineState.CANCELLED

class ElementStreamTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.cache = InMemoryCache()
        self.p = beam.Pipeline()
        self.pcoll = self.p | beam.Create([])
        self.cache_key = str(CacheKey('pcoll', '', '', ''))
        self.mock_result = MockPipelineResult()
        ie.current_env().add_user_pipeline(self.p)
        ie.current_env().set_pipeline_result(self.p, self.mock_result)
        ie.current_env().set_cache_manager(self.cache, self.p)

    def test_read(self):
        if False:
            for i in range(10):
                print('nop')
        'Test reading and if a stream is done no more elements are returned.'
        self.mock_result.set_state(PipelineState.DONE)
        self.cache.write(['expected'], 'full', self.cache_key)
        self.cache.save_pcoder(None, 'full', self.cache_key)
        stream = ElementStream(self.pcoll, '', self.cache_key, max_n=1, max_duration_secs=1)
        self.assertFalse(stream.is_done())
        self.assertEqual(list(stream.read())[0], 'expected')
        self.assertTrue(stream.is_done())

    def test_done_if_terminated(self):
        if False:
            print('Hello World!')
        'Test that terminating the job sets the stream as done.'
        self.cache.write(['expected'], 'full', self.cache_key)
        self.cache.save_pcoder(None, 'full', self.cache_key)
        stream = ElementStream(self.pcoll, '', self.cache_key, max_n=100, max_duration_secs=10)
        self.assertFalse(stream.is_done())
        self.assertEqual(list(stream.read(tail=False))[0], 'expected')
        self.assertFalse(stream.is_done())
        self.mock_result.set_state(PipelineState.DONE)
        self.assertEqual(list(stream.read(tail=False))[0], 'expected')
        self.assertTrue(stream.is_done())

    def test_read_n(self):
        if False:
            while True:
                i = 10
        "Test that the stream only reads 'n' elements."
        self.mock_result.set_state(PipelineState.DONE)
        self.cache.write(list(range(5)), 'full', self.cache_key)
        self.cache.save_pcoder(None, 'full', self.cache_key)
        stream = ElementStream(self.pcoll, '', self.cache_key, max_n=1, max_duration_secs=1)
        self.assertEqual(list(stream.read()), [0])
        self.assertTrue(stream.is_done())
        stream = ElementStream(self.pcoll, '', self.cache_key, max_n=2, max_duration_secs=1)
        self.assertEqual(list(stream.read()), [0, 1])
        self.assertTrue(stream.is_done())
        stream = ElementStream(self.pcoll, '', self.cache_key, max_n=5, max_duration_secs=1)
        self.assertEqual(list(stream.read()), list(range(5)))
        self.assertTrue(stream.is_done())
        stream = ElementStream(self.pcoll, '', self.cache_key, max_n=10, max_duration_secs=1)
        self.assertEqual(list(stream.read()), list(range(5)))
        self.assertTrue(stream.is_done())

    def test_read_duration(self):
        if False:
            while True:
                i = 10
        "Test that the stream only reads a 'duration' of elements."

        def as_windowed_value(element):
            if False:
                while True:
                    i = 10
            return WindowedValueHolder(WindowedValue(element, 0, []))
        values = FileRecordsBuilder(tag=self.cache_key).advance_processing_time(1).add_element(element=as_windowed_value(0), event_time_secs=0).advance_processing_time(1).add_element(element=as_windowed_value(1), event_time_secs=1).advance_processing_time(1).add_element(element=as_windowed_value(2), event_time_secs=3).advance_processing_time(1).add_element(element=as_windowed_value(3), event_time_secs=4).advance_processing_time(1).add_element(element=as_windowed_value(4), event_time_secs=5).build()
        values = [v.recorded_event for v in values if isinstance(v, beam_interactive_api_pb2.TestStreamFileRecord)]
        self.mock_result.set_state(PipelineState.DONE)
        self.cache.write(values, 'full', self.cache_key)
        self.cache.save_pcoder(coders.FastPrimitivesCoder(), 'full', self.cache_key)
        stream = ElementStream(self.pcoll, '', self.cache_key, max_n=100, max_duration_secs=1)
        self.assertSequenceEqual([e.value for e in stream.read()], [0])
        stream = ElementStream(self.pcoll, '', self.cache_key, max_n=100, max_duration_secs=2)
        self.assertSequenceEqual([e.value for e in stream.read()], [0, 1])
        stream = ElementStream(self.pcoll, '', self.cache_key, max_n=100, max_duration_secs=10)
        self.assertSequenceEqual([e.value for e in stream.read()], [0, 1, 2, 3, 4])

class RecordingTest(unittest.TestCase):

    def test_computed(self):
        if False:
            while True:
                i = 10
        'Tests that a PCollection is marked as computed only in a complete state.\n\n    Because the background caching job is now long-lived, repeated runs of a\n    PipelineFragment may yield different results for the same PCollection.\n    '
        p = beam.Pipeline(InteractiveRunner())
        elems = p | beam.Create([0, 1, 2])
        ib.watch(locals())
        mock_result = MockPipelineResult()
        ie.current_env().track_user_pipelines()
        ie.current_env().set_pipeline_result(p, mock_result)
        bcj_mock_result = MockPipelineResult()
        background_caching_job = bcj.BackgroundCachingJob(bcj_mock_result, [])
        recording = Recording(p, [elems], mock_result, max_n=10, max_duration_secs=60)
        self.assertFalse(recording.is_computed())
        self.assertFalse(recording.computed())
        self.assertTrue(recording.uncomputed())
        mock_result.set_state(PipelineState.DONE)
        recording.wait_until_finish()
        self.assertFalse(recording.is_computed())
        self.assertFalse(recording.computed())
        self.assertTrue(recording.uncomputed())
        bcj_mock_result.set_state(PipelineState.DONE)
        ie.current_env().set_background_caching_job(p, background_caching_job)
        recording = Recording(p, [elems], mock_result, max_n=10, max_duration_secs=60)
        recording.wait_until_finish()
        self.assertTrue(recording.is_computed())
        self.assertTrue(recording.computed())
        self.assertFalse(recording.uncomputed())

    def test_describe(self):
        if False:
            return 10
        p = beam.Pipeline(InteractiveRunner())
        numbers = p | 'numbers' >> beam.Create([0, 1, 2])
        letters = p | 'letters' >> beam.Create(['a', 'b', 'c'])
        ib.watch(locals())
        mock_result = MockPipelineResult()
        ie.current_env().track_user_pipelines()
        ie.current_env().set_pipeline_result(p, mock_result)
        cache_manager = InMemoryCache()
        ie.current_env().set_cache_manager(cache_manager, p)
        recording = Recording(p, [numbers, letters], mock_result, max_n=10, max_duration_secs=60)
        numbers_stream = recording.stream(numbers)
        cache_manager.write([0, 1, 2], 'full', numbers_stream.cache_key)
        cache_manager.save_pcoder(None, 'full', numbers_stream.cache_key)
        letters_stream = recording.stream(letters)
        cache_manager.write(['a', 'b', 'c'], 'full', letters_stream.cache_key)
        cache_manager.save_pcoder(None, 'full', letters_stream.cache_key)
        description = recording.describe()
        size = description['size']
        self.assertEqual(size, cache_manager.size('full', numbers_stream.cache_key) + cache_manager.size('full', letters_stream.cache_key))

class RecordingManagerTest(unittest.TestCase):

    def test_basic_execution(self):
        if False:
            i = 10
            return i + 15
        'A basic pipeline to be used as a smoke test.'
        p = beam.Pipeline(InteractiveRunner())
        numbers = p | 'numbers' >> beam.Create([0, 1, 2])
        letters = p | 'letters' >> beam.Create(['a', 'b', 'c'])
        ib.watch(locals())
        ie.current_env().track_user_pipelines()
        rm = RecordingManager(p)
        numbers_recording = rm.record([numbers], max_n=3, max_duration=500)
        numbers_stream = numbers_recording.stream(numbers)
        numbers_recording.wait_until_finish()
        elems = list(numbers_stream.read())
        expected_elems = [WindowedValue(i, MIN_TIMESTAMP, [GlobalWindow()]) for i in range(3)]
        self.assertListEqual(elems, expected_elems)
        letters_recording = rm.record([letters], max_n=3, max_duration=500)
        letters_recording.wait_until_finish()
        self.assertEqual(rm.describe()['size'], numbers_recording.describe()['size'] + letters_recording.describe()['size'])
        rm.cancel()

    def test_duration_parsing(self):
        if False:
            return 10
        p = beam.Pipeline(InteractiveRunner())
        elems = p | beam.Create([0, 1, 2])
        ib.watch(locals())
        ie.current_env().track_user_pipelines()
        rm = RecordingManager(p)
        recording = rm.record([elems], max_n=3, max_duration='500s')
        recording.wait_until_finish()
        self.assertEqual(recording.describe()['duration'], 500)

    def test_cancel_stops_recording(self):
        if False:
            while True:
                i = 10
        ib.options.recordable_sources.add(TestStream)
        p = beam.Pipeline(InteractiveRunner(), options=PipelineOptions(streaming=True))
        elems = p | TestStream().advance_watermark_to(0).advance_processing_time(1).add_elements(list(range(10))).advance_processing_time(1)
        squares = elems | beam.Map(lambda x: x ** 2)
        ib.watch(locals())
        ie.current_env().track_user_pipelines()

        class SemaphoreLimiter(Limiter):

            def __init__(self):
                if False:
                    return 10
                self.triggered = False

            def is_triggered(self):
                if False:
                    i = 10
                    return i + 15
                return self.triggered
        semaphore_limiter = SemaphoreLimiter()
        rm = RecordingManager(p, test_limiters=[semaphore_limiter])
        rm.record([squares], max_n=10, max_duration=500)
        bcj = ie.current_env().get_background_caching_job(p)
        self.assertFalse(bcj.is_done())
        semaphore_limiter.triggered = True
        rm.cancel()
        self.assertTrue(bcj.is_done())

    def test_recording_manager_clears_cache(self):
        if False:
            i = 10
            return i + 15
        'Tests that the RecordingManager clears the cache before recording.\n\n    A job may have incomplete PCollections when the job terminates. Clearing the\n    cache ensures that correct results are computed every run.\n    '
        ib.options.recordable_sources.add(TestStream)
        p = beam.Pipeline(InteractiveRunner(), options=PipelineOptions(streaming=True))
        elems = p | TestStream().advance_watermark_to(0).advance_processing_time(1).add_elements(list(range(10))).advance_processing_time(1)
        squares = elems | beam.Map(lambda x: x ** 2)
        ib.watch(locals())
        ie.current_env().track_user_pipelines()
        rm = RecordingManager(p)
        rm._clear_pcolls = MagicMock()
        rm.record([squares], max_n=1, max_duration=500)
        rm.cancel()
        rm._clear_pcolls.assert_any_call(unittest.mock.ANY, {CacheKey.from_pcoll('squares', squares).to_str()})

    def test_clear(self):
        if False:
            return 10
        p1 = beam.Pipeline(InteractiveRunner())
        elems_1 = p1 | 'elems 1' >> beam.Create([0, 1, 2])
        ib.watch(locals())
        ie.current_env().track_user_pipelines()
        recording_manager = RecordingManager(p1)
        recording = recording_manager.record([elems_1], max_n=3, max_duration=500)
        recording.wait_until_finish()
        record_describe = recording_manager.describe()
        self.assertGreater(record_describe['size'], 0)
        recording_manager.clear()
        self.assertEqual(recording_manager.describe()['size'], 0)

    def test_clear_specific_pipeline(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that clear can empty the cache for a specific pipeline.'
        p1 = beam.Pipeline(InteractiveRunner())
        elems_1 = p1 | 'elems 1' >> beam.Create([0, 1, 2])
        p2 = beam.Pipeline(InteractiveRunner())
        elems_2 = p2 | 'elems 2' >> beam.Create([0, 1, 2])
        ib.watch(locals())
        ie.current_env().track_user_pipelines()
        rm_1 = RecordingManager(p1)
        recording = rm_1.record([elems_1], max_n=3, max_duration=500)
        recording.wait_until_finish()
        rm_2 = RecordingManager(p2)
        recording = rm_2.record([elems_2], max_n=3, max_duration=500)
        recording.wait_until_finish()
        if rm_1.describe()['state'] == PipelineState.STOPPED and rm_2.describe()['state'] == PipelineState.STOPPED:
            self.assertGreater(rm_1.describe()['size'], 0)
            self.assertGreater(rm_2.describe()['size'], 0)
            rm_1.clear()
            self.assertEqual(rm_1.describe()['size'], 0)
            self.assertGreater(rm_2.describe()['size'], 0)
            rm_2.clear()
            self.assertEqual(rm_2.describe()['size'], 0)

    def test_record_pipeline(self):
        if False:
            return 10
        ib.options.recordable_sources.add(TestStream)
        p = beam.Pipeline(InteractiveRunner(), options=PipelineOptions(streaming=True))
        _ = p | TestStream().advance_watermark_to(0).advance_processing_time(1).add_elements(list(range(10))).advance_processing_time(1)
        ib.watch(locals())
        ie.current_env().track_user_pipelines()

        class SizeLimiter(Limiter):

            def __init__(self, p):
                if False:
                    print('Hello World!')
                self.pipeline = p
                self._rm = None

            def set_recording_manager(self, rm):
                if False:
                    while True:
                        i = 10
                self._rm = rm

            def is_triggered(self):
                if False:
                    i = 10
                    return i + 15
                return self._rm.describe()['size'] > 0 if self._rm else False
        size_limiter = SizeLimiter(p)
        rm = RecordingManager(p, test_limiters=[size_limiter])
        size_limiter.set_recording_manager(rm)
        self.assertEqual(rm.describe()['state'], PipelineState.STOPPED)
        self.assertTrue(rm.record_pipeline())
        self.assertFalse(rm.record_pipeline())
        for _ in range(60):
            if rm.describe()['state'] == PipelineState.CANCELLED:
                break
            time.sleep(1)
        self.assertTrue(rm.describe()['state'] == PipelineState.CANCELLED, 'Test timed out waiting for pipeline to be cancelled. This indicates that the BackgroundCachingJob did not cache anything.')

    @patch('apache_beam.runners.interactive.recording_manager.RecordingManager._clear_pcolls', return_value=None)
    @patch('apache_beam.runners.interactive.pipeline_fragment.PipelineFragment.run', return_value=None)
    def test_record_detects_remote_runner(self, mock_pipeline_fragment, mock_clear_pcolls):
        if False:
            while True:
                i = 10
        'Tests that a remote runner is detected, resulting in the\n    PipelineFragment instance to have blocking enabled.'
        p = beam.Pipeline(InteractiveRunner())
        numbers = p | 'numbers' >> beam.Create([0, 1, 2])
        ib.options.cache_root = 'gs://test-bucket/'
        rm = RecordingManager(p)
        rm.record([numbers], max_n=3, max_duration=500)
        mock_pipeline_fragment.assert_called_with(blocking=True)
        ib.options.cache_root = None
if __name__ == '__main__':
    unittest.main()