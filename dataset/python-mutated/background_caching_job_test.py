"""Tests for apache_beam.runners.interactive.background_caching_job."""
import unittest
from unittest.mock import patch
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.pipeline import PipelineVisitor
from apache_beam.runners import runner
from apache_beam.runners.interactive import background_caching_job as bcj
from apache_beam.runners.interactive import interactive_beam as ib
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive import interactive_runner
from apache_beam.runners.interactive.caching.streaming_cache import StreamingCache
from apache_beam.runners.interactive.testing.mock_ipython import mock_get_ipython
from apache_beam.runners.interactive.testing.test_cache_manager import FileRecordsBuilder
from apache_beam.testing.test_stream import TestStream
from apache_beam.testing.test_stream_service import TestStreamServiceController
from apache_beam.transforms.window import TimestampedValue
_FOO_PUBSUB_SUB = 'projects/test-project/subscriptions/foo'
_BAR_PUBSUB_SUB = 'projects/test-project/subscriptions/bar'
_TEST_CACHE_KEY = 'test'

def _build_a_test_stream_pipeline():
    if False:
        print('Hello World!')
    test_stream = TestStream().advance_watermark_to(0).add_elements([TimestampedValue('a', 1)]).advance_processing_time(5).advance_watermark_to_infinity()
    p = beam.Pipeline(runner=interactive_runner.InteractiveRunner())
    events = p | test_stream
    ib.watch(locals())
    return p

def _build_an_empty_stream_pipeline():
    if False:
        print('Hello World!')
    pipeline_options = PipelineOptions(streaming=True)
    p = beam.Pipeline(interactive_runner.InteractiveRunner(), options=pipeline_options)
    ib.watch({'pipeline': p})
    return p

def _setup_test_streaming_cache(pipeline):
    if False:
        i = 10
        return i + 15
    cache_manager = StreamingCache(cache_dir=None)
    ie.current_env().set_cache_manager(cache_manager, pipeline)
    builder = FileRecordsBuilder(tag=_TEST_CACHE_KEY)
    builder.advance_watermark(watermark_secs=0).advance_processing_time(5).add_element(element='a', event_time_secs=1).advance_watermark(watermark_secs=100).advance_processing_time(10)
    cache_manager.write(builder.build(), _TEST_CACHE_KEY)

@unittest.skipIf(not ie.current_env().is_interactive_ready, '[interactive] dependency is not installed.')
class BackgroundCachingJobTest(unittest.TestCase):

    def tearDown(self):
        if False:
            return 10
        ie.new_env()

    @patch('apache_beam.runners.interactive.background_caching_job.has_source_to_cache', lambda x: True)
    @patch('apache_beam.runners.interactive.interactive_environment.InteractiveEnvironment.cleanup', lambda x, y: None)
    def test_background_caching_job_starts_when_none_such_job_exists(self):
        if False:
            for i in range(10):
                print('nop')

        class FakePipelineResult(beam.runners.runner.PipelineResult):

            def wait_until_finish(self):
                if False:
                    i = 10
                    return i + 15
                return

        class FakePipelineRunner(beam.runners.PipelineRunner):

            def run_pipeline(self, pipeline, options):
                if False:
                    return 10
                return FakePipelineResult(beam.runners.runner.PipelineState.RUNNING)
        p = beam.Pipeline(runner=interactive_runner.InteractiveRunner(FakePipelineRunner()), options=PipelineOptions(streaming=True))
        elems = p | 'Read' >> beam.io.ReadFromPubSub(subscription=_FOO_PUBSUB_SUB)
        ib.watch(locals())
        _setup_test_streaming_cache(p)
        p.run()
        self.assertIsNotNone(ie.current_env().get_background_caching_job(p))
        expected_cached_source_signature = bcj.extract_source_to_cache_signature(p)
        self.assertEqual(expected_cached_source_signature, ie.current_env().get_cached_source_signature(p))

    @patch('apache_beam.runners.interactive.background_caching_job.has_source_to_cache', lambda x: False)
    def test_background_caching_job_not_start_for_batch_pipeline(self):
        if False:
            i = 10
            return i + 15
        p = beam.Pipeline()
        p | beam.Create([])
        p.run()
        self.assertIsNone(ie.current_env().get_background_caching_job(p))

    @patch('apache_beam.runners.interactive.background_caching_job.has_source_to_cache', lambda x: True)
    @patch('apache_beam.runners.interactive.interactive_environment.InteractiveEnvironment.cleanup', lambda x, y: None)
    def test_background_caching_job_not_start_when_such_job_exists(self):
        if False:
            return 10
        p = _build_a_test_stream_pipeline()
        _setup_test_streaming_cache(p)
        a_running_background_caching_job = bcj.BackgroundCachingJob(runner.PipelineResult(runner.PipelineState.RUNNING), limiters=[])
        ie.current_env().set_background_caching_job(p, a_running_background_caching_job)
        main_job_result = p.run()
        self.assertIs(a_running_background_caching_job, ie.current_env().get_background_caching_job(p))
        self.assertIs(main_job_result, ie.current_env().pipeline_result(p))

    @patch('apache_beam.runners.interactive.background_caching_job.has_source_to_cache', lambda x: True)
    @patch('apache_beam.runners.interactive.interactive_environment.InteractiveEnvironment.cleanup', lambda x, y: None)
    def test_background_caching_job_not_start_when_such_job_is_done(self):
        if False:
            for i in range(10):
                print('nop')
        p = _build_a_test_stream_pipeline()
        _setup_test_streaming_cache(p)
        a_done_background_caching_job = bcj.BackgroundCachingJob(runner.PipelineResult(runner.PipelineState.DONE), limiters=[])
        ie.current_env().set_background_caching_job(p, a_done_background_caching_job)
        main_job_result = p.run()
        self.assertIs(a_done_background_caching_job, ie.current_env().get_background_caching_job(p))
        self.assertIs(main_job_result, ie.current_env().pipeline_result(p))

    @patch('IPython.get_ipython', new_callable=mock_get_ipython)
    def test_source_to_cache_changed_when_pipeline_is_first_time_seen(self, cell):
        if False:
            return 10
        with cell:
            pipeline = _build_an_empty_stream_pipeline()
        with cell:
            read_foo = pipeline | 'Read' >> beam.io.ReadFromPubSub(subscription=_FOO_PUBSUB_SUB)
            ib.watch({'read_foo': read_foo})
        self.assertTrue(bcj.is_source_to_cache_changed(pipeline))

    @patch('IPython.get_ipython', new_callable=mock_get_ipython)
    def test_source_to_cache_changed_when_new_source_is_added(self, cell):
        if False:
            i = 10
            return i + 15
        with cell:
            pipeline = _build_an_empty_stream_pipeline()
            read_foo = pipeline | 'Read' >> beam.io.ReadFromPubSub(subscription=_FOO_PUBSUB_SUB)
            ib.watch({'read_foo': read_foo})
        ie.current_env().set_cached_source_signature(pipeline, bcj.extract_source_to_cache_signature(pipeline))
        self.assertFalse(bcj.is_cache_complete(str(id(pipeline))))
        with cell:
            read_bar = pipeline | 'Read' >> beam.io.ReadFromPubSub(subscription=_BAR_PUBSUB_SUB)
            ib.watch({'read_bar': read_bar})
        self.assertTrue(bcj.is_cache_complete(str(id(pipeline))))
        self.assertTrue(bcj.is_source_to_cache_changed(pipeline))

    @patch('IPython.get_ipython', new_callable=mock_get_ipython)
    def test_source_to_cache_changed_when_source_is_altered(self, cell):
        if False:
            print('Hello World!')
        with cell:
            pipeline = _build_an_empty_stream_pipeline()
            transform = beam.io.ReadFromPubSub(subscription=_FOO_PUBSUB_SUB)
            read_foo = pipeline | 'Read' >> transform
            ib.watch({'read_foo': read_foo})
        ie.current_env().set_cached_source_signature(pipeline, bcj.extract_source_to_cache_signature(pipeline))
        with cell:
            from apache_beam.io.gcp.pubsub import _PubSubSource
            transform._source = _PubSubSource(subscription=_BAR_PUBSUB_SUB)
        self.assertTrue(bcj.is_source_to_cache_changed(pipeline))

    @patch('IPython.get_ipython', new_callable=mock_get_ipython)
    def test_source_to_cache_not_changed_for_same_source(self, cell):
        if False:
            print('Hello World!')
        with cell:
            pipeline = _build_an_empty_stream_pipeline()
            transform = beam.io.ReadFromPubSub(subscription=_FOO_PUBSUB_SUB)
        with cell:
            read_foo_1 = pipeline | 'Read' >> transform
            ib.watch({'read_foo_1': read_foo_1})
        ie.current_env().set_cached_source_signature(pipeline, bcj.extract_source_to_cache_signature(pipeline))
        with cell:
            read_foo_2 = pipeline | 'Read' >> transform
            ib.watch({'read_foo_2': read_foo_2})
        self.assertFalse(bcj.is_source_to_cache_changed(pipeline))
        with cell:
            read_foo_3 = pipeline | 'Read' >> beam.io.ReadFromPubSub(subscription=_FOO_PUBSUB_SUB)
            ib.watch({'read_foo_3': read_foo_3})
        self.assertFalse(bcj.is_source_to_cache_changed(pipeline))

    @patch('IPython.get_ipython', new_callable=mock_get_ipython)
    def test_source_to_cache_not_changed_when_source_is_removed(self, cell):
        if False:
            return 10
        with cell:
            pipeline = _build_an_empty_stream_pipeline()
            foo_transform = beam.io.ReadFromPubSub(subscription=_FOO_PUBSUB_SUB)
            bar_transform = beam.io.ReadFromPubSub(subscription=_BAR_PUBSUB_SUB)
        with cell:
            read_foo = pipeline | 'Read' >> foo_transform
            ib.watch({'read_foo': read_foo})
        signature_with_only_foo = bcj.extract_source_to_cache_signature(pipeline)
        with cell:
            read_bar = pipeline | 'Read' >> bar_transform
            ib.watch({'read_bar': read_bar})
        self.assertTrue(bcj.is_source_to_cache_changed(pipeline))
        signature_with_foo_bar = ie.current_env().get_cached_source_signature(pipeline)
        self.assertNotEqual(signature_with_only_foo, signature_with_foo_bar)

        class BarPruneVisitor(PipelineVisitor):

            def enter_composite_transform(self, transform_node):
                if False:
                    return 10
                pruned_parts = list(transform_node.parts)
                for part in transform_node.parts:
                    if part.transform is bar_transform:
                        pruned_parts.remove(part)
                transform_node.parts = tuple(pruned_parts)
                self.visit_transform(transform_node)

            def visit_transform(self, transform_node):
                if False:
                    while True:
                        i = 10
                if transform_node.transform is bar_transform:
                    transform_node.parent = None
        v = BarPruneVisitor()
        pipeline.visit(v)
        signature_after_pruning_bar = bcj.extract_source_to_cache_signature(pipeline)
        self.assertEqual(signature_with_only_foo, signature_after_pruning_bar)
        self.assertFalse(bcj.is_source_to_cache_changed(pipeline))

    def test_determine_a_test_stream_service_running(self):
        if False:
            i = 10
            return i + 15
        pipeline = _build_an_empty_stream_pipeline()
        test_stream_service = TestStreamServiceController(reader=None)
        test_stream_service.start()
        ie.current_env().set_test_stream_service_controller(pipeline, test_stream_service)
        self.assertTrue(bcj.is_a_test_stream_service_running(pipeline))

    def test_stop_a_running_test_stream_service(self):
        if False:
            i = 10
            return i + 15
        pipeline = _build_an_empty_stream_pipeline()
        test_stream_service = TestStreamServiceController(reader=None)
        test_stream_service.start()
        ie.current_env().set_test_stream_service_controller(pipeline, test_stream_service)
        bcj.attempt_to_stop_test_stream_service(pipeline)
        self.assertFalse(bcj.is_a_test_stream_service_running(pipeline))

    @patch('apache_beam.testing.test_stream_service.TestStreamServiceController.stop')
    def test_noop_when_no_test_stream_service_running(self, _mocked_stop):
        if False:
            print('Hello World!')
        pipeline = _build_an_empty_stream_pipeline()
        self.assertFalse(bcj.is_a_test_stream_service_running(pipeline))
        bcj.attempt_to_stop_test_stream_service(pipeline)
        _mocked_stop.assert_not_called()
if __name__ == '__main__':
    unittest.main()