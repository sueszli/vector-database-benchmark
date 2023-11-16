import logging
import os
import shutil
import tempfile
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from google.protobuf.message import DecodeError
import apache_beam as beam
from apache_beam import coders
from apache_beam.portability.api import beam_interactive_api_pb2
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.runners.interactive.cache_manager import CacheManager
from apache_beam.runners.interactive.cache_manager import SafeFastPrimitivesCoder
from apache_beam.runners.interactive.caching.cacheable import CacheKey
from apache_beam.testing.test_stream import OutputFormat
from apache_beam.testing.test_stream import ReverseTestStream
from apache_beam.utils import timestamp
_LOGGER = logging.getLogger(__name__)

class StreamingCacheSink(beam.PTransform):
    """A PTransform that writes TestStreamFile(Header|Records)s to file.

  This transform takes in an arbitrary element stream and writes the list of
  TestStream events (as TestStreamFileRecords) to file. When replayed, this
  will produce the best-effort replay of the original job (e.g. some elements
  may be produced slightly out of order from the original stream).

  Note that this PTransform is assumed to be only run on a single machine where
  the following assumptions are correct: elements come in ordered, no two
  transforms are writing to the same file. This PTransform is assumed to only
  run correctly with the DirectRunner.

  TODO(https://github.com/apache/beam/issues/20002): Generalize this to more
  source/sink types aside from file based. Also, generalize to cases where
  there might be multiple workers writing to the same sink.
  """

    def __init__(self, cache_dir, filename, sample_resolution_sec, coder=SafeFastPrimitivesCoder()):
        if False:
            return 10
        self._cache_dir = cache_dir
        self._filename = filename
        self._sample_resolution_sec = sample_resolution_sec
        self._coder = coder
        self._path = os.path.join(self._cache_dir, self._filename)

    @property
    def path(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the path the sink leads to.'
        return self._path

    @property
    def size_in_bytes(self):
        if False:
            return 10
        'Returns the space usage in bytes of the sink.'
        try:
            return os.stat(self._path).st_size
        except OSError:
            _LOGGER.debug('Failed to calculate cache size for file %s, the file might have not been created yet. Return 0. %s', self._path, traceback.format_exc())
            return 0

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')

        class StreamingWriteToText(beam.DoFn):
            """DoFn that performs the writing.

      Note that the other file writing methods cannot be used in streaming
      contexts.
      """

            def __init__(self, full_path, coder=SafeFastPrimitivesCoder()):
                if False:
                    while True:
                        i = 10
                self._full_path = full_path
                self._coder = coder
                Path(os.path.dirname(full_path)).mkdir(parents=True, exist_ok=True)

            def start_bundle(self):
                if False:
                    while True:
                        i = 10
                self._fh = open(self._full_path, 'ab')

            def finish_bundle(self):
                if False:
                    print('Hello World!')
                self._fh.close()

            def process(self, e):
                if False:
                    print('Hello World!')
                'Appends the given element to the file.\n        '
                self._fh.write(self._coder.encode(e) + b'\n')
        return pcoll | ReverseTestStream(output_tag=self._filename, sample_resolution_sec=self._sample_resolution_sec, output_format=OutputFormat.SERIALIZED_TEST_STREAM_FILE_RECORDS, coder=self._coder) | beam.ParDo(StreamingWriteToText(full_path=self._path, coder=self._coder))

class StreamingCacheSource:
    """A class that reads and parses TestStreamFile(Header|Reader)s.

  This source operates in the following way:

    1. Wait for up to `timeout_secs` for the file to be available.
    2. Read, parse, and emit the entire contents of the file
    3. Wait for more events to come or until `is_cache_complete` returns True
    4. If there are more events, then go to 2
    5. Otherwise, stop emitting.

  This class is used to read from file and send its to the TestStream via the
  StreamingCacheManager.Reader.
  """

    def __init__(self, cache_dir, labels, is_cache_complete=None, coder=None):
        if False:
            return 10
        if not coder:
            coder = SafeFastPrimitivesCoder()
        if not is_cache_complete:
            is_cache_complete = lambda _: True
        self._cache_dir = cache_dir
        self._coder = coder
        self._labels = labels
        self._path = os.path.join(self._cache_dir, *self._labels)
        self._is_cache_complete = is_cache_complete
        self._pipeline_id = CacheKey.from_str(labels[-1]).pipeline_id

    def _wait_until_file_exists(self, timeout_secs=30):
        if False:
            return 10
        'Blocks until the file exists for a maximum of timeout_secs.\n    '
        start = time.time()
        while not os.path.exists(self._path):
            time.sleep(1)
            if time.time() - start > timeout_secs:
                pcollection_var = CacheKey.from_str(self._labels[-1]).var
                raise RuntimeError('Timed out waiting for cache file for PCollection `{}` to be available with path {}.'.format(pcollection_var, self._path))
        return open(self._path, mode='rb')

    def _emit_from_file(self, fh, tail):
        if False:
            for i in range(10):
                print('nop')
        'Emits the TestStreamFile(Header|Record)s from file.\n\n    This returns a generator to be able to read all lines from the given file.\n    If `tail` is True, then it will wait until the cache is complete to exit.\n    Otherwise, it will read the file only once.\n    '
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line or (line and line[-1] != b'\n'[0]):
                if not tail and pos != 0:
                    break
                if self._is_cache_complete(self._pipeline_id):
                    break
                time.sleep(0.5)
                fh.seek(pos)
            else:
                to_decode = line[:-1]
                if pos == 0:
                    proto_cls = beam_interactive_api_pb2.TestStreamFileHeader
                else:
                    proto_cls = beam_interactive_api_pb2.TestStreamFileRecord
                msg = self._try_parse_as(proto_cls, to_decode)
                if msg:
                    yield msg
                else:
                    break

    def _try_parse_as(self, proto_cls, to_decode):
        if False:
            while True:
                i = 10
        try:
            msg = proto_cls()
            msg.ParseFromString(self._coder.decode(to_decode))
        except DecodeError:
            _LOGGER.error('Could not parse as %s. This can indicate that the cache is corruputed. Please restart the kernel. \nfile: %s \nmessage: %s', proto_cls, self._path, to_decode)
            msg = None
        return msg

    def read(self, tail):
        if False:
            print('Hello World!')
        'Reads all TestStreamFile(Header|TestStreamFileRecord)s from file.\n\n    This returns a generator to be able to read all lines from the given file.\n    If `tail` is True, then it will wait until the cache is complete to exit.\n    Otherwise, it will read the file only once.\n    '
        with self._wait_until_file_exists() as f:
            for e in self._emit_from_file(f, tail):
                yield e

class StreamingCache(CacheManager):
    """Abstraction that holds the logic for reading and writing to cache.
  """

    def __init__(self, cache_dir, is_cache_complete=None, sample_resolution_sec=0.1, saved_pcoders=None):
        if False:
            i = 10
            return i + 15
        self._sample_resolution_sec = sample_resolution_sec
        self._is_cache_complete = is_cache_complete
        if cache_dir:
            self._cache_dir = cache_dir
        else:
            self._cache_dir = tempfile.mkdtemp(prefix='ib-', dir=os.environ.get('TEST_TMPDIR', None))
        self._saved_pcoders = saved_pcoders or {}
        self._default_pcoder = SafeFastPrimitivesCoder()
        self._capture_sinks = {}
        self._capture_keys = set()

    def size(self, *labels):
        if False:
            for i in range(10):
                print('nop')
        if self.exists(*labels):
            return os.path.getsize(os.path.join(self._cache_dir, *labels))
        return 0

    @property
    def capture_size(self):
        if False:
            print('Hello World!')
        return sum([sink.size_in_bytes for (_, sink) in self._capture_sinks.items()])

    @property
    def capture_paths(self):
        if False:
            while True:
                i = 10
        return list(self._capture_sinks.keys())

    @property
    def capture_keys(self):
        if False:
            i = 10
            return i + 15
        return self._capture_keys

    def exists(self, *labels):
        if False:
            return 10
        if labels and any(labels):
            path = os.path.join(self._cache_dir, *labels)
            return os.path.exists(path)
        return False

    def read(self, *labels, **args):
        if False:
            i = 10
            return i + 15
        'Returns a generator to read all records from file.'
        tail = args.pop('tail', False)
        if not self.exists(*labels) and (not tail):
            return (iter([]), -1)
        reader = StreamingCacheSource(self._cache_dir, labels, self._is_cache_complete, self.load_pcoder(*labels)).read(tail=tail)
        try:
            header = next(reader)
        except StopIteration:
            return (iter([]), -1)
        return (StreamingCache.Reader([header], [reader]).read(), 1)

    def read_multiple(self, labels, tail=True):
        if False:
            while True:
                i = 10
        'Returns a generator to read all records from file.\n\n    Does tail until the cache is complete. This is because it is used in the\n    TestStreamServiceController to read from file which is only used during\n    pipeline runtime which needs to block.\n    '
        readers = [StreamingCacheSource(self._cache_dir, l, self._is_cache_complete, self.load_pcoder(*l)).read(tail=tail) for l in labels]
        headers = [next(r) for r in readers]
        return StreamingCache.Reader(headers, readers).read()

    def write(self, values, *labels):
        if False:
            while True:
                i = 10
        'Writes the given values to cache.\n    '
        directory = os.path.join(self._cache_dir, *labels[:-1])
        filepath = os.path.join(directory, labels[-1])
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filepath, 'ab') as f:
            for v in values:
                if isinstance(v, (beam_interactive_api_pb2.TestStreamFileHeader, beam_interactive_api_pb2.TestStreamFileRecord)):
                    val = v.SerializeToString()
                else:
                    raise TypeError('Values given to streaming cache should be either TestStreamFileHeader or TestStreamFileRecord.')
                f.write(self.load_pcoder(*labels).encode(val) + b'\n')

    def clear(self, *labels):
        if False:
            for i in range(10):
                print('nop')
        directory = os.path.join(self._cache_dir, *labels[:-1])
        filepath = os.path.join(directory, labels[-1])
        self._capture_keys.discard(labels[-1])
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    def source(self, *labels):
        if False:
            return 10
        'Returns the StreamingCacheManager source.\n\n    This is beam.Impulse() because unbounded sources will be marked with this\n    and then the PipelineInstrument will replace these with a TestStream.\n    '
        return beam.Impulse()

    def sink(self, labels, is_capture=False):
        if False:
            for i in range(10):
                print('nop')
        'Returns a StreamingCacheSink to write elements to file.\n\n    Note that this is assumed to only work in the DirectRunner as the underlying\n    StreamingCacheSink assumes a single machine to have correct element\n    ordering.\n    '
        filename = labels[-1]
        cache_dir = os.path.join(self._cache_dir, *labels[:-1])
        sink = StreamingCacheSink(cache_dir, filename, self._sample_resolution_sec, self.load_pcoder(*labels))
        if is_capture:
            self._capture_sinks[sink.path] = sink
            self._capture_keys.add(filename)
        return sink

    def save_pcoder(self, pcoder, *labels):
        if False:
            return 10
        self._saved_pcoders[os.path.join(self._cache_dir, *labels)] = pcoder

    def load_pcoder(self, *labels):
        if False:
            print('Hello World!')
        saved_pcoder = self._saved_pcoders.get(os.path.join(self._cache_dir, *labels), None)
        if saved_pcoder is None or isinstance(saved_pcoder, coders.FastPrimitivesCoder):
            return self._default_pcoder
        return saved_pcoder

    def cleanup(self):
        if False:
            while True:
                i = 10
        if os.path.exists(self._cache_dir):

            def on_fail_to_cleanup(function, path, excinfo):
                if False:
                    for i in range(10):
                        print('nop')
                _LOGGER.warning('Failed to clean up temporary files: %s. You maymanually delete them if necessary. Error was: %s', path, excinfo)
            shutil.rmtree(self._cache_dir, onerror=on_fail_to_cleanup)
        self._saved_pcoders = {}
        self._capture_sinks = {}
        self._capture_keys = set()

    class Reader(object):
        """Abstraction that reads from PCollection readers.

    This class is an Abstraction layer over multiple PCollection readers to be
    used for supplying a TestStream service with events.

    This class is also responsible for holding the state of the clock, injecting
    clock advancement events, and watermark advancement events.
    """

        def __init__(self, headers, readers):
            if False:
                for i in range(10):
                    print('nop')
            self._monotonic_clock = timestamp.Timestamp.of(0)
            self._readers = {}
            self._headers = {header.tag: header for header in headers}
            self._readers = OrderedDict(((h.tag, r) for (h, r) in zip(headers, readers)))
            self._stream_times = {tag: timestamp.Timestamp(seconds=0) for tag in self._headers}

        def _test_stream_events_before_target(self, target_timestamp):
            if False:
                print('Hello World!')
            'Reads the next iteration of elements from each stream.\n\n      Retrieves an element from each stream iff the most recently read timestamp\n      from that stream is less than the target_timestamp. Since the amount of\n      events may not fit into memory, this StreamingCache reads at most one\n      element from each stream at a time.\n      '
            records = []
            for (tag, r) in self._readers.items():
                if self._stream_times[tag] >= target_timestamp:
                    continue
                try:
                    record = next(r).recorded_event
                    if record.HasField('processing_time_event'):
                        self._stream_times[tag] += timestamp.Duration(micros=record.processing_time_event.advance_duration)
                    records.append((tag, record, self._stream_times[tag]))
                except StopIteration:
                    pass
            return records

        def _merge_sort(self, previous_events, new_events):
            if False:
                while True:
                    i = 10
            return sorted(previous_events + new_events, key=lambda x: x[2], reverse=True)

        def _min_timestamp_of(self, events):
            if False:
                while True:
                    i = 10
            return events[-1][2] if events else timestamp.MAX_TIMESTAMP

        def _event_stream_caught_up_to_target(self, events, target_timestamp):
            if False:
                print('Hello World!')
            empty_events = not events
            stream_is_past_target = self._min_timestamp_of(events) > target_timestamp
            return empty_events or stream_is_past_target

        def read(self):
            if False:
                for i in range(10):
                    print('nop')
            'Reads records from PCollection readers.\n      '
            target_timestamp = timestamp.MAX_TIMESTAMP
            unsent_events = []
            while True:
                new_events = self._test_stream_events_before_target(target_timestamp)
                events_to_send = self._merge_sort(unsent_events, new_events)
                if not events_to_send:
                    break
                target_timestamp = self._min_timestamp_of(events_to_send)
                while not self._event_stream_caught_up_to_target(events_to_send, target_timestamp):
                    (tag, r, curr_timestamp) = events_to_send.pop()
                    if curr_timestamp > self._monotonic_clock:
                        yield self._advance_processing_time(curr_timestamp)
                    if r.HasField('element_event'):
                        r.element_event.tag = tag
                        yield r
                    elif r.HasField('watermark_event'):
                        r.watermark_event.tag = tag
                        yield r
                unsent_events = events_to_send
                target_timestamp = self._min_timestamp_of(unsent_events)

        def _advance_processing_time(self, new_timestamp):
            if False:
                i = 10
                return i + 15
            'Advances the internal clock and returns an AdvanceProcessingTime event.\n      '
            advancy_by = new_timestamp.micros - self._monotonic_clock.micros
            e = beam_runner_api_pb2.TestStreamPayload.Event(processing_time_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceProcessingTime(advance_duration=advancy_by))
            self._monotonic_clock = new_timestamp
            return e