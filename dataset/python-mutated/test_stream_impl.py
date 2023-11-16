"""The TestStream implementation for the DirectRunner

The DirectRunner implements TestStream as the _TestStream class which is used
to store the events in memory, the _WatermarkController which is used to set the
watermark and emit events, and the multiplexer which sends events to the correct
tagged PCollection.
"""
import itertools
import logging
from queue import Empty as EmptyException
from queue import Queue
from threading import Thread
import grpc
from apache_beam import ParDo
from apache_beam import coders
from apache_beam import pvalue
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.portability.api import beam_runner_api_pb2_grpc
from apache_beam.testing.test_stream import ElementEvent
from apache_beam.testing.test_stream import ProcessingTimeEvent
from apache_beam.testing.test_stream import WatermarkEvent
from apache_beam.transforms import PTransform
from apache_beam.transforms import core
from apache_beam.transforms import window
from apache_beam.transforms.window import TimestampedValue
from apache_beam.utils import timestamp
from apache_beam.utils.timestamp import Duration
from apache_beam.utils.timestamp import Timestamp
_LOGGER = logging.getLogger(__name__)

class _EndOfStream:
    pass

class _WatermarkController(PTransform):
    """A runner-overridable PTransform Primitive to control the watermark.

  Expected implementation behavior:
   - If the instance recieves a WatermarkEvent, it sets its output watermark to
     the specified value then drops the event.
   - If the instance receives an ElementEvent, it emits all specified elements
     to the Global Window with the event time set to the element's timestamp.
  """

    def __init__(self, output_tag):
        if False:
            i = 10
            return i + 15
        self.output_tag = output_tag

    def get_windowing(self, _):
        if False:
            i = 10
            return i + 15
        return core.Windowing(window.GlobalWindows())

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        ret = pvalue.PCollection.from_(pcoll)
        ret.tag = self.output_tag
        return ret

class _ExpandableTestStream(PTransform):

    def __init__(self, test_stream):
        if False:
            return 10
        self.test_stream = test_stream

    def expand(self, pbegin):
        if False:
            for i in range(10):
                print('nop')
        'Expands the TestStream into the DirectRunner implementation.\n\n    Takes the TestStream transform and creates a _TestStream -> multiplexer ->\n    _WatermarkController.\n    '
        assert isinstance(pbegin, pvalue.PBegin)
        if len(self.test_stream.output_tags) == 1:
            return pbegin | _TestStream(self.test_stream.output_tags, events=self.test_stream._events, coder=self.test_stream.coder, endpoint=self.test_stream._endpoint) | _WatermarkController(list(self.test_stream.output_tags)[0])

        def mux(event):
            if False:
                i = 10
                return i + 15
            if event.tag:
                yield pvalue.TaggedOutput(event.tag, event)
            else:
                yield event
        mux_output = pbegin | _TestStream(self.test_stream.output_tags, events=self.test_stream._events, coder=self.test_stream.coder, endpoint=self.test_stream._endpoint) | 'TestStream Multiplexer' >> ParDo(mux).with_outputs()
        outputs = {}
        for tag in self.test_stream.output_tags:
            label = '_WatermarkController[{}]'.format(tag)
            outputs[tag] = mux_output[tag] | label >> _WatermarkController(tag)
        return outputs

class _TestStream(PTransform):
    """Test stream that generates events on an unbounded PCollection of elements.

  Each event emits elements, advances the watermark or advances the processing
  time.  After all of the specified elements are emitted, ceases to produce
  output.

  Expected implementation behavior:
   - If the instance receives a WatermarkEvent with the WATERMARK_CONTROL_TAG
     then the instance sets its own watermark hold at the specified value and
     drops the event.
   - If the instance receives any other WatermarkEvent or ElementEvent, it
     passes it to the consumer.
  """
    WATERMARK_CONTROL_TAG = '_TestStream_Watermark'

    def __init__(self, output_tags, coder=coders.FastPrimitivesCoder(), events=None, endpoint=None):
        if False:
            print('Hello World!')
        assert coder is not None
        self.coder = coder
        self._raw_events = events
        self._events = self._add_watermark_advancements(output_tags, events)
        self.output_tags = output_tags
        self.endpoint = endpoint

    def _watermark_starts(self, output_tags):
        if False:
            while True:
                i = 10
        'Sentinel values to hold the watermark of outputs to -inf.\n\n    The output watermarks of the output PCollections (fake unbounded sources) in\n    a TestStream are controlled by watermark holds. This sets the hold of each\n    output PCollection so that the individual holds can be controlled by the\n    given events.\n    '
        return [WatermarkEvent(timestamp.MIN_TIMESTAMP, tag) for tag in output_tags]

    def _watermark_stops(self, output_tags):
        if False:
            i = 10
            return i + 15
        'Sentinel values to close the watermark of outputs.'
        return [WatermarkEvent(timestamp.MAX_TIMESTAMP, tag) for tag in output_tags]

    def _test_stream_start(self):
        if False:
            return 10
        'Sentinel value to move the watermark hold of the TestStream to +inf.\n\n    This sets a hold to +inf such that the individual holds of the output\n    PCollections are allowed to modify their individial output watermarks with\n    their holds. This is because the calculation of the output watermark is a\n    min over all input watermarks.\n    '
        return [WatermarkEvent(timestamp.MAX_TIMESTAMP - timestamp.TIME_GRANULARITY, _TestStream.WATERMARK_CONTROL_TAG)]

    def _test_stream_stop(self):
        if False:
            return 10
        'Sentinel value to close the watermark of the TestStream.'
        return [WatermarkEvent(timestamp.MAX_TIMESTAMP, _TestStream.WATERMARK_CONTROL_TAG)]

    def _test_stream_init(self):
        if False:
            i = 10
            return i + 15
        'Sentinel value to hold the watermark of the TestStream to -inf.\n\n    This sets a hold to ensure that the output watermarks of the output\n    PCollections do not advance to +inf before their watermark holds are set.\n    '
        return [WatermarkEvent(timestamp.MIN_TIMESTAMP, _TestStream.WATERMARK_CONTROL_TAG)]

    def _set_up(self, output_tags):
        if False:
            i = 10
            return i + 15
        return self._test_stream_init() + self._watermark_starts(output_tags) + self._test_stream_start()

    def _tear_down(self, output_tags):
        if False:
            print('Hello World!')
        return self._watermark_stops(output_tags) + self._test_stream_stop()

    def _add_watermark_advancements(self, output_tags, events):
        if False:
            return 10
        'Adds watermark advancements to the given events.\n\n    The following watermark advancements can be done on the runner side.\n    However, it makes the logic on the runner side much more complicated than\n    it needs to be.\n\n    In order for watermarks to be properly advanced in a TestStream, a specific\n    sequence of watermark holds must be sent:\n\n    1. Hold the root watermark at -inf (this prevents the pipeline from\n       immediately returning).\n    2. Hold the watermarks at the WatermarkControllerss at -inf (this prevents\n       the pipeline from immediately returning).\n    3. Advance the root watermark to +inf - 1 (this allows the downstream\n       WatermarkControllers to control their watermarks via holds).\n    4. Advance watermarks as normal.\n    5. Advance WatermarkController watermarks to +inf\n    6. Advance root watermark to +inf.\n    '
        if not events:
            return []
        return self._set_up(output_tags) + events + self._tear_down(output_tags)

    def get_windowing(self, unused_inputs):
        if False:
            i = 10
            return i + 15
        return core.Windowing(window.GlobalWindows())

    def expand(self, pcoll):
        if False:
            i = 10
            return i + 15
        return pvalue.PCollection(pcoll.pipeline, is_bounded=False)

    def _infer_output_coder(self, input_type=None, input_coder=None):
        if False:
            for i in range(10):
                print('nop')
        return self.coder

    @staticmethod
    def events_from_script(events):
        if False:
            print('Hello World!')
        'Yields the in-memory events.\n    '
        return itertools.chain(events)

    @staticmethod
    def _stream_events_from_rpc(endpoint, output_tags, coder, channel, is_alive):
        if False:
            while True:
                i = 10
        'Yields the events received from the given endpoint.\n\n    This is the producer thread that reads events from the TestStreamService and\n    puts them onto the shared queue. At the end of the stream, an _EndOfStream\n    is placed on the channel to signify a successful end.\n    '
        stub_channel = grpc.insecure_channel(endpoint)
        stub = beam_runner_api_pb2_grpc.TestStreamServiceStub(stub_channel)
        event_request = beam_runner_api_pb2.EventsRequest(output_ids=[str(tag) for tag in output_tags])
        event_stream = stub.Events(event_request)
        try:
            for e in event_stream:
                channel.put(_TestStream.test_stream_payload_to_events(e, coder))
                if not is_alive():
                    return
        except grpc.RpcError as e:
            if e.code() in (grpc.StatusCode.CANCELLED, grpc.StatusCode.UNAVAILABLE):
                return
            raise e
        finally:
            channel.put(_EndOfStream())

    @staticmethod
    def events_from_rpc(endpoint, output_tags, coder, evaluation_context):
        if False:
            print('Hello World!')
        'Yields the events received from the given endpoint.\n\n    This method starts a new thread that reads from the TestStreamService and\n    puts the events onto a shared queue. This method then yields all elements\n    from the queue. Unfortunately, this is necessary because the GRPC API does\n    not allow for non-blocking calls when utilizing a streaming RPC. It is\n    officially suggested from the docs to use a producer/consumer pattern to\n    handle streaming RPCs. By doing so, this gives this method control over when\n    to cancel reading from the RPC if the server takes too long to respond.\n    '
        shutdown_requested = False

        def is_alive():
            if False:
                return 10
            return not (shutdown_requested or evaluation_context.shutdown_requested)
        channel = Queue()
        event_stream = Thread(target=_TestStream._stream_events_from_rpc, args=(endpoint, output_tags, coder, channel, is_alive))
        event_stream.setDaemon(True)
        event_stream.start()
        while True:
            try:
                event = channel.get(timeout=30)
                if isinstance(event, _EndOfStream):
                    break
                yield event
            except EmptyException as e:
                _LOGGER.warning('TestStream timed out waiting for new events from service. Stopping pipeline.')
                shutdown_requested = True
                raise e

    @staticmethod
    def test_stream_payload_to_events(payload, coder):
        if False:
            i = 10
            return i + 15
        'Returns a TestStream Python event object from a TestStream event Proto.\n    '
        if payload.HasField('element_event'):
            element_event = payload.element_event
            elements = [TimestampedValue(coder.decode(e.encoded_element), Timestamp(micros=e.timestamp)) for e in element_event.elements]
            return ElementEvent(timestamped_values=elements, tag=element_event.tag)
        if payload.HasField('watermark_event'):
            watermark_event = payload.watermark_event
            return WatermarkEvent(Timestamp(micros=watermark_event.new_watermark), tag=watermark_event.tag)
        if payload.HasField('processing_time_event'):
            processing_time_event = payload.processing_time_event
            return ProcessingTimeEvent(Duration(micros=processing_time_event.advance_duration))
        raise RuntimeError('Received a proto without the specified fields: {}'.format(payload))