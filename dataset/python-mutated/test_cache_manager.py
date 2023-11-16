import collections
import itertools
import sys
import apache_beam as beam
from apache_beam import coders
from apache_beam.portability.api import beam_interactive_api_pb2
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.runners.interactive.cache_manager import CacheManager
from apache_beam.utils.timestamp import Duration
from apache_beam.utils.timestamp import Timestamp

class InMemoryCache(CacheManager):
    """A cache that stores all PCollections in an in-memory map.

  This is only used for checking the pipeline shape. This can't be used for
  running the pipeline isn't shared between the SDK and the Runner.
  """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._cached = {}
        self._pcoders = {}

    def exists(self, *labels):
        if False:
            for i in range(10):
                print('nop')
        return self._key(*labels) in self._cached

    def _latest_version(self, *labels):
        if False:
            i = 10
            return i + 15
        return True

    def read(self, *labels, **args):
        if False:
            print('Hello World!')
        if not self.exists(*labels):
            return (itertools.chain([]), -1)
        return (itertools.chain(self._cached[self._key(*labels)]), None)

    def write(self, value, *labels):
        if False:
            for i in range(10):
                print('nop')
        if not self.exists(*labels):
            self._cached[self._key(*labels)] = []
        self._cached[self._key(*labels)] += value

    def save_pcoder(self, pcoder, *labels):
        if False:
            i = 10
            return i + 15
        self._pcoders[self._key(*labels)] = pcoder

    def load_pcoder(self, *labels):
        if False:
            i = 10
            return i + 15
        return self._pcoders[self._key(*labels)]

    def cleanup(self):
        if False:
            while True:
                i = 10
        self._cached = collections.defaultdict(list)
        self._pcoders = {}

    def clear(self, *label):
        if False:
            return 10
        pass

    def source(self, *labels):
        if False:
            while True:
                i = 10
        vals = self._cached[self._key(*labels)]
        return beam.Create(vals)

    def sink(self, labels, is_capture=False):
        if False:
            print('Hello World!')
        return beam.Map(lambda _: _)

    def size(self, *labels):
        if False:
            while True:
                i = 10
        if self.exists(*labels):
            return sys.getsizeof(self._cached[self._key(*labels)])
        return 0

    def _key(self, *labels):
        if False:
            for i in range(10):
                print('nop')
        return '/'.join([l for l in labels])

class NoopSink(beam.PTransform):

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        return pcoll | beam.Map(lambda x: x)

class FileRecordsBuilder(object):

    def __init__(self, tag=None):
        if False:
            i = 10
            return i + 15
        self._header = beam_interactive_api_pb2.TestStreamFileHeader(tag=tag)
        self._records = []
        self._coder = coders.FastPrimitivesCoder()

    def add_element(self, element, event_time_secs):
        if False:
            print('Hello World!')
        element_payload = beam_runner_api_pb2.TestStreamPayload.TimestampedElement(encoded_element=self._coder.encode(element), timestamp=Timestamp.of(event_time_secs).micros)
        record = beam_interactive_api_pb2.TestStreamFileRecord(recorded_event=beam_runner_api_pb2.TestStreamPayload.Event(element_event=beam_runner_api_pb2.TestStreamPayload.Event.AddElements(elements=[element_payload])))
        self._records.append(record)
        return self

    def advance_watermark(self, watermark_secs):
        if False:
            while True:
                i = 10
        record = beam_interactive_api_pb2.TestStreamFileRecord(recorded_event=beam_runner_api_pb2.TestStreamPayload.Event(watermark_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceWatermark(new_watermark=Timestamp.of(watermark_secs).micros)))
        self._records.append(record)
        return self

    def advance_processing_time(self, delta_secs):
        if False:
            while True:
                i = 10
        record = beam_interactive_api_pb2.TestStreamFileRecord(recorded_event=beam_runner_api_pb2.TestStreamPayload.Event(processing_time_event=beam_runner_api_pb2.TestStreamPayload.Event.AdvanceProcessingTime(advance_duration=Duration.of(delta_secs).micros)))
        self._records.append(record)
        return self

    def build(self):
        if False:
            for i in range(10):
                print('nop')
        return [self._header] + self._records