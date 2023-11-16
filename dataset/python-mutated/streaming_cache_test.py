import unittest
from apache_beam import coders
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.portability.api import beam_interactive_api_pb2
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.runners.interactive import interactive_beam as ib
from apache_beam.runners.interactive.cache_manager import SafeFastPrimitivesCoder
from apache_beam.runners.interactive.caching.cacheable import CacheKey
from apache_beam.runners.interactive.caching.streaming_cache import StreamingCache
from apache_beam.runners.interactive.testing.test_cache_manager import FileRecordsBuilder
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.test_stream import TestStream
from apache_beam.testing.util import *
from apache_beam.transforms.window import TimestampedValue
beam_runner_api_pb2.TestStreamPayload.__test__ = False
beam_interactive_api_pb2.TestStreamFileHeader.__test__ = False
beam_interactive_api_pb2.TestStreamFileRecord.__test__ = False

class StreamingCacheTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        pass

    def test_exists(self):
        if False:
            i = 10
            return i + 15
        cache = StreamingCache(cache_dir=None)
        self.assertFalse(cache.exists('my_label'))
        cache.write([beam_interactive_api_pb2.TestStreamFileRecord()], 'my_label')
        self.assertTrue(cache.exists('my_label'))
        self.assertFalse(cache.exists(''))

    def test_empty(self):
        if False:
            while True:
                i = 10
        CACHED_PCOLLECTION_KEY = repr(CacheKey('arbitrary_key', '', '', ''))
        cache = StreamingCache(cache_dir=None)
        self.assertFalse(cache.exists(CACHED_PCOLLECTION_KEY))
        cache.write([], CACHED_PCOLLECTION_KEY)
        (reader, _) = cache.read(CACHED_PCOLLECTION_KEY)
        self.assertFalse([e for e in reader])

    def test_size(self):
        if False:
            i = 10
            return i + 15
        cache = StreamingCache(cache_dir=None)
        cache.write([beam_interactive_api_pb2.TestStreamFileRecord()], 'my_label')
        coder = cache.load_pcoder('my_label')
        size = len(coder.encode(beam_interactive_api_pb2.TestStreamFileRecord().SerializeToString())) + 1
        self.assertEqual(cache.size('my_label'), size)

    def test_clear(self):
        if False:
            print('Hello World!')
        cache = StreamingCache(cache_dir=None)
        self.assertFalse(cache.exists('my_label'))
        cache.sink(['my_label'], is_capture=True)
        cache.write([beam_interactive_api_pb2.TestStreamFileRecord()], 'my_label')
        self.assertTrue(cache.exists('my_label'))
        self.assertEqual(cache.capture_keys, set(['my_label']))
        self.assertTrue(cache.clear('my_label'))
        self.assertFalse(cache.exists('my_label'))
        self.assertFalse(cache.capture_keys)

    def test_single_reader(self):
        if False:
            print('Hello World!')
        '\n    Tests that we expect to see all the correctly emitted TestStreamPayloads.\n    '
        CACHED_PCOLLECTION_KEY = repr(CacheKey('arbitrary_key', '', '', ''))
        values = FileRecordsBuilder(tag=CACHED_PCOLLECTION_KEY).add_element(element=0, event_time_secs=0).advance_processing_time(1).add_element(element=1, event_time_secs=1).advance_processing_time(1).add_element(element=2, event_time_secs=2).build()
        cache = StreamingCache(cache_dir=None)
        cache.write(values, CACHED_PCOLLECTION_KEY)
        (reader, _) = cache.read(CACHED_PCOLLECTION_KEY)
        coder = coders.FastPrimitivesCoder()
        events = list(reader)
        expected = [beam_runner_api_pb2.TestStreamPayload.Event(element_event=beam_runner_api_pb2.TestStreamPayload.Event.AddElements(elements=[beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode(0), timestamp=0)], tag=CACHED_PCOLLECTION_KEY)), beam_runner_api_pb2.TestStreamPayload.Event(processing_time_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceProcessingTime(advance_duration=1 * 10 ** 6)), beam_runner_api_pb2.TestStreamPayload.Event(element_event=beam_runner_api_pb2.TestStreamPayload.Event.AddElements(elements=[beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode(1), timestamp=1 * 10 ** 6)], tag=CACHED_PCOLLECTION_KEY)), beam_runner_api_pb2.TestStreamPayload.Event(processing_time_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceProcessingTime(advance_duration=1 * 10 ** 6)), beam_runner_api_pb2.TestStreamPayload.Event(element_event=beam_runner_api_pb2.TestStreamPayload.Event.AddElements(elements=[beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode(2), timestamp=2 * 10 ** 6)], tag=CACHED_PCOLLECTION_KEY))]
        self.assertSequenceEqual(events, expected)

    def test_multiple_readers(self):
        if False:
            while True:
                i = 10
        'Tests that the service advances the clock with multiple outputs.\n    '
        CACHED_LETTERS = repr(CacheKey('letters', '', '', ''))
        CACHED_NUMBERS = repr(CacheKey('numbers', '', '', ''))
        CACHED_LATE = repr(CacheKey('late', '', '', ''))
        letters = FileRecordsBuilder(CACHED_LETTERS).advance_processing_time(1).advance_watermark(watermark_secs=0).add_element(element='a', event_time_secs=0).advance_processing_time(10).advance_watermark(watermark_secs=10).add_element(element='b', event_time_secs=10).build()
        numbers = FileRecordsBuilder(CACHED_NUMBERS).advance_processing_time(2).add_element(element=1, event_time_secs=0).advance_processing_time(1).add_element(element=2, event_time_secs=0).advance_processing_time(1).add_element(element=2, event_time_secs=0).build()
        late = FileRecordsBuilder(CACHED_LATE).advance_processing_time(101).add_element(element='late', event_time_secs=0).build()
        cache = StreamingCache(cache_dir=None)
        cache.write(letters, CACHED_LETTERS)
        cache.write(numbers, CACHED_NUMBERS)
        cache.write(late, CACHED_LATE)
        reader = cache.read_multiple([[CACHED_LETTERS], [CACHED_NUMBERS], [CACHED_LATE]])
        coder = coders.FastPrimitivesCoder()
        events = list(reader)
        expected = [beam_runner_api_pb2.TestStreamPayload.Event(processing_time_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceProcessingTime(advance_duration=1 * 10 ** 6)), beam_runner_api_pb2.TestStreamPayload.Event(watermark_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceWatermark(new_watermark=0, tag=CACHED_LETTERS)), beam_runner_api_pb2.TestStreamPayload.Event(element_event=beam_runner_api_pb2.TestStreamPayload.Event.AddElements(elements=[beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode('a'), timestamp=0)], tag=CACHED_LETTERS)), beam_runner_api_pb2.TestStreamPayload.Event(processing_time_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceProcessingTime(advance_duration=1 * 10 ** 6)), beam_runner_api_pb2.TestStreamPayload.Event(element_event=beam_runner_api_pb2.TestStreamPayload.Event.AddElements(elements=[beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode(1), timestamp=0)], tag=CACHED_NUMBERS)), beam_runner_api_pb2.TestStreamPayload.Event(processing_time_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceProcessingTime(advance_duration=1 * 10 ** 6)), beam_runner_api_pb2.TestStreamPayload.Event(element_event=beam_runner_api_pb2.TestStreamPayload.Event.AddElements(elements=[beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode(2), timestamp=0)], tag=CACHED_NUMBERS)), beam_runner_api_pb2.TestStreamPayload.Event(processing_time_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceProcessingTime(advance_duration=1 * 10 ** 6)), beam_runner_api_pb2.TestStreamPayload.Event(element_event=beam_runner_api_pb2.TestStreamPayload.Event.AddElements(elements=[beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode(2), timestamp=0)], tag=CACHED_NUMBERS)), beam_runner_api_pb2.TestStreamPayload.Event(processing_time_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceProcessingTime(advance_duration=7 * 10 ** 6)), beam_runner_api_pb2.TestStreamPayload.Event(watermark_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceWatermark(new_watermark=10 * 10 ** 6, tag=CACHED_LETTERS)), beam_runner_api_pb2.TestStreamPayload.Event(element_event=beam_runner_api_pb2.TestStreamPayload.Event.AddElements(elements=[beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode('b'), timestamp=10 * 10 ** 6)], tag=CACHED_LETTERS)), beam_runner_api_pb2.TestStreamPayload.Event(processing_time_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceProcessingTime(advance_duration=90 * 10 ** 6)), beam_runner_api_pb2.TestStreamPayload.Event(element_event=beam_runner_api_pb2.TestStreamPayload.Event.AddElements(elements=[beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode('late'), timestamp=0)], tag=CACHED_LATE))]
        self.assertSequenceEqual(events, expected)

    def test_read_and_write(self):
        if False:
            print('Hello World!')
        'An integration test between the Sink and Source.\n\n    This ensures that the sink and source speak the same language in terms of\n    coders, protos, order, and units.\n    '
        CACHED_RECORDS = repr(CacheKey('records', '', '', ''))
        test_stream = TestStream(output_tags=CACHED_RECORDS).advance_watermark_to(0, tag=CACHED_RECORDS).advance_processing_time(5).add_elements(['a', 'b', 'c'], tag=CACHED_RECORDS).advance_watermark_to(10, tag=CACHED_RECORDS).advance_processing_time(1).add_elements([TimestampedValue('1', 15), TimestampedValue('2', 15), TimestampedValue('3', 15)], tag=CACHED_RECORDS)
        coder = SafeFastPrimitivesCoder()
        cache = StreamingCache(cache_dir=None, sample_resolution_sec=1.0)
        self.assertEqual(cache.capture_keys, set())
        options = StandardOptions(streaming=True)
        with TestPipeline(options=options) as p:
            records = (p | test_stream)[CACHED_RECORDS]
            records | cache.sink([CACHED_RECORDS], is_capture=True)
        (reader, _) = cache.read(CACHED_RECORDS)
        actual_events = list(reader)
        self.assertEqual(cache.capture_keys, set([CACHED_RECORDS]))
        expected_events = [beam_runner_api_pb2.TestStreamPayload.Event(processing_time_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceProcessingTime(advance_duration=5 * 10 ** 6)), beam_runner_api_pb2.TestStreamPayload.Event(watermark_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceWatermark(new_watermark=0, tag=CACHED_RECORDS)), beam_runner_api_pb2.TestStreamPayload.Event(element_event=beam_runner_api_pb2.TestStreamPayload.Event.AddElements(elements=[beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode('a'), timestamp=0), beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode('b'), timestamp=0), beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode('c'), timestamp=0)], tag=CACHED_RECORDS)), beam_runner_api_pb2.TestStreamPayload.Event(processing_time_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceProcessingTime(advance_duration=1 * 10 ** 6)), beam_runner_api_pb2.TestStreamPayload.Event(watermark_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceWatermark(new_watermark=10 * 10 ** 6, tag=CACHED_RECORDS)), beam_runner_api_pb2.TestStreamPayload.Event(element_event=beam_runner_api_pb2.TestStreamPayload.Event.AddElements(elements=[beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode('1'), timestamp=15 * 10 ** 6), beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode('2'), timestamp=15 * 10 ** 6), beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode('3'), timestamp=15 * 10 ** 6)], tag=CACHED_RECORDS))]
        self.assertEqual(actual_events, expected_events)

    def test_read_and_write_multiple_outputs(self):
        if False:
            i = 10
            return i + 15
        'An integration test between the Sink and Source with multiple outputs.\n\n    This tests the funcionatlity that the StreamingCache reads from multiple\n    files and combines them into a single sorted output.\n    '
        LETTERS_TAG = repr(CacheKey('letters', '', '', ''))
        NUMBERS_TAG = repr(CacheKey('numbers', '', '', ''))
        test_stream = TestStream().advance_watermark_to(0, tag=LETTERS_TAG).advance_processing_time(5).add_elements(['a', 'b', 'c'], tag=LETTERS_TAG).advance_watermark_to(10, tag=NUMBERS_TAG).advance_processing_time(1).add_elements([TimestampedValue('1', 15), TimestampedValue('2', 15), TimestampedValue('3', 15)], tag=NUMBERS_TAG)
        cache = StreamingCache(cache_dir=None, sample_resolution_sec=1.0)
        coder = SafeFastPrimitivesCoder()
        options = StandardOptions(streaming=True)
        with TestPipeline(options=options) as p:
            events = p | test_stream
            events[LETTERS_TAG] | 'Letters sink' >> cache.sink([LETTERS_TAG])
            events[NUMBERS_TAG] | 'Numbers sink' >> cache.sink([NUMBERS_TAG])
        reader = cache.read_multiple([[LETTERS_TAG], [NUMBERS_TAG]])
        actual_events = list(reader)
        expected_events = [beam_runner_api_pb2.TestStreamPayload.Event(processing_time_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceProcessingTime(advance_duration=5 * 10 ** 6)), beam_runner_api_pb2.TestStreamPayload.Event(watermark_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceWatermark(new_watermark=0, tag=LETTERS_TAG)), beam_runner_api_pb2.TestStreamPayload.Event(element_event=beam_runner_api_pb2.TestStreamPayload.Event.AddElements(elements=[beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode('a'), timestamp=0), beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode('b'), timestamp=0), beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode('c'), timestamp=0)], tag=LETTERS_TAG)), beam_runner_api_pb2.TestStreamPayload.Event(processing_time_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceProcessingTime(advance_duration=1 * 10 ** 6)), beam_runner_api_pb2.TestStreamPayload.Event(watermark_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceWatermark(new_watermark=10 * 10 ** 6, tag=NUMBERS_TAG)), beam_runner_api_pb2.TestStreamPayload.Event(watermark_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceWatermark(new_watermark=0, tag=LETTERS_TAG)), beam_runner_api_pb2.TestStreamPayload.Event(element_event=beam_runner_api_pb2.TestStreamPayload.Event.AddElements(elements=[beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode('1'), timestamp=15 * 10 ** 6), beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode('2'), timestamp=15 * 10 ** 6), beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=coder.encode('3'), timestamp=15 * 10 ** 6)], tag=NUMBERS_TAG))]
        self.assertListEqual(actual_events, expected_events)

    def test_always_default_coder_for_test_stream_records(self):
        if False:
            print('Hello World!')
        CACHED_NUMBERS = repr(CacheKey('numbers', '', '', ''))
        numbers = FileRecordsBuilder(CACHED_NUMBERS).advance_processing_time(2).add_element(element=1, event_time_secs=0).advance_processing_time(1).add_element(element=2, event_time_secs=0).advance_processing_time(1).add_element(element=2, event_time_secs=0).build()
        cache = StreamingCache(cache_dir=None)
        cache.write(numbers, CACHED_NUMBERS)
        self.assertIs(type(cache.load_pcoder(CACHED_NUMBERS)), type(cache._default_pcoder))

    def test_streaming_cache_does_not_write_non_record_or_header_types(self):
        if False:
            return 10
        cache = StreamingCache(cache_dir=None)
        self.assertRaises(TypeError, cache.write, 'some value', 'a key')

    def test_streaming_cache_uses_gcs_ib_cache_root(self):
        if False:
            while True:
                i = 10
        '\n    Checks that StreamingCache._cache_dir is set to the\n    cache_root set under Interactive Beam for a GCS directory.\n    '
        ib.options.cache_root = 'gs://'
        cache_manager_with_ib_option = StreamingCache(cache_dir=ib.options.cache_root)
        self.assertEqual(ib.options.cache_root, cache_manager_with_ib_option._cache_dir)
        ib.options.cache_root = None

    def test_streaming_cache_uses_local_ib_cache_root(self):
        if False:
            i = 10
            return i + 15
        '\n    Checks that StreamingCache._cache_dir is set to the\n    cache_root set under Interactive Beam for a local directory\n    and that the cached values are the same as the values of a\n    cache using default settings.\n    '
        CACHED_PCOLLECTION_KEY = repr(CacheKey('arbitrary_key', '', '', ''))
        values = FileRecordsBuilder(CACHED_PCOLLECTION_KEY).advance_processing_time(1).advance_watermark(watermark_secs=0).add_element(element=1, event_time_secs=0).build()
        local_cache = StreamingCache(cache_dir=None)
        local_cache.write(values, CACHED_PCOLLECTION_KEY)
        (reader_one, _) = local_cache.read(CACHED_PCOLLECTION_KEY)
        pcoll_list_one = list(reader_one)
        ib.options.cache_root = '/tmp/it-test/'
        cache_manager_with_ib_option = StreamingCache(cache_dir=ib.options.cache_root)
        self.assertEqual(ib.options.cache_root, cache_manager_with_ib_option._cache_dir)
        cache_manager_with_ib_option.write(values, CACHED_PCOLLECTION_KEY)
        (reader_two, _) = cache_manager_with_ib_option.read(CACHED_PCOLLECTION_KEY)
        pcoll_list_two = list(reader_two)
        self.assertEqual(pcoll_list_one, pcoll_list_two)
        ib.options.cache_root = None
if __name__ == '__main__':
    unittest.main()