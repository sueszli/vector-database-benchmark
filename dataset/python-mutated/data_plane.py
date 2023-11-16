"""Implementation of ``DataChannel``s to communicate across the data plane."""
import abc
import collections
import json
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Collection
from typing import DefaultDict
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union
import grpc
from apache_beam.coders import coder_impl
from apache_beam.portability.api import beam_fn_api_pb2
from apache_beam.portability.api import beam_fn_api_pb2_grpc
from apache_beam.runners.worker.channel_factory import GRPCChannelFactory
from apache_beam.runners.worker.worker_id_interceptor import WorkerIdInterceptor
if TYPE_CHECKING:
    import apache_beam.coders.slow_stream
    OutputStream = apache_beam.coders.slow_stream.OutputStream
    DataOrTimers = Union[beam_fn_api_pb2.Elements.Data, beam_fn_api_pb2.Elements.Timers]
else:
    OutputStream = type(coder_impl.create_OutputStream())
_LOGGER = logging.getLogger(__name__)
_DEFAULT_SIZE_FLUSH_THRESHOLD = 10 << 20
_DEFAULT_TIME_FLUSH_THRESHOLD_MS = 0
_MAX_CLEANED_INSTRUCTIONS = 10000
_GRPC_SERVICE_CONFIG = json.dumps({'methodConfig': [{'name': [{'service': 'org.apache.beam.model.fn_execution.v1.BeamFnData'}], 'retryPolicy': {'maxAttempts': 5, 'initialBackoff': '0.1s', 'maxBackoff': '5s', 'backoffMultiplier': 2, 'retryableStatusCodes': ['UNAVAILABLE']}}]})

class ClosableOutputStream(OutputStream):
    """A Outputstream for use with CoderImpls that has a close() method."""

    def __init__(self, close_callback=None):
        if False:
            print('Hello World!')
        super().__init__()
        self._close_callback = close_callback

    def close(self):
        if False:
            while True:
                i = 10
        if self._close_callback:
            self._close_callback(self.get())

    def maybe_flush(self):
        if False:
            while True:
                i = 10
        pass

    def flush(self):
        if False:
            i = 10
            return i + 15
        pass

    @staticmethod
    def create(close_callback, flush_callback, data_buffer_time_limit_ms):
        if False:
            return 10
        if data_buffer_time_limit_ms > 0:
            return TimeBasedBufferingClosableOutputStream(close_callback, flush_callback=flush_callback, time_flush_threshold_ms=data_buffer_time_limit_ms)
        else:
            return SizeBasedBufferingClosableOutputStream(close_callback, flush_callback=flush_callback)

class SizeBasedBufferingClosableOutputStream(ClosableOutputStream):
    """A size-based buffering OutputStream."""

    def __init__(self, close_callback=None, flush_callback=None, size_flush_threshold=_DEFAULT_SIZE_FLUSH_THRESHOLD):
        if False:
            print('Hello World!')
        super().__init__(close_callback)
        self._flush_callback = flush_callback
        self._size_flush_threshold = size_flush_threshold

    def maybe_flush(self):
        if False:
            return 10
        if self.size() > self._size_flush_threshold:
            self.flush()

    def flush(self):
        if False:
            print('Hello World!')
        if self._flush_callback:
            self._flush_callback(self.get())
            self._clear()

class TimeBasedBufferingClosableOutputStream(SizeBasedBufferingClosableOutputStream):
    """A buffering OutputStream with both time-based and size-based."""
    _periodic_flusher = None

    def __init__(self, close_callback=None, flush_callback=None, size_flush_threshold=_DEFAULT_SIZE_FLUSH_THRESHOLD, time_flush_threshold_ms=_DEFAULT_TIME_FLUSH_THRESHOLD_MS):
        if False:
            return 10
        super().__init__(close_callback, flush_callback, size_flush_threshold)
        assert time_flush_threshold_ms > 0
        self._time_flush_threshold_ms = time_flush_threshold_ms
        self._flush_lock = threading.Lock()
        self._schedule_lock = threading.Lock()
        self._closed = False
        self._schedule_periodic_flush()

    def flush(self):
        if False:
            return 10
        with self._flush_lock:
            super().flush()

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        with self._schedule_lock:
            self._closed = True
            if self._periodic_flusher:
                self._periodic_flusher.cancel()
                self._periodic_flusher = None
        super().close()

    def _schedule_periodic_flush(self):
        if False:
            for i in range(10):
                print('nop')

        def _flush():
            if False:
                i = 10
                return i + 15
            with self._schedule_lock:
                if not self._closed:
                    self.flush()
        self._periodic_flusher = PeriodicThread(self._time_flush_threshold_ms / 1000.0, _flush)
        self._periodic_flusher.daemon = True
        self._periodic_flusher.start()

class PeriodicThread(threading.Thread):
    """Call a function periodically with the specified number of seconds"""

    def __init__(self, interval, function, args=None, kwargs=None):
        if False:
            print('Hello World!')
        threading.Thread.__init__(self)
        self._interval = interval
        self._function = function
        self._args = args if args is not None else []
        self._kwargs = kwargs if kwargs is not None else {}
        self._finished = threading.Event()

    def run(self):
        if False:
            return 10
        next_call = time.time() + self._interval
        while not self._finished.wait(next_call - time.time()):
            next_call = next_call + self._interval
            self._function(*self._args, **self._kwargs)

    def cancel(self):
        if False:
            while True:
                i = 10
        "Stop the thread if it hasn't finished yet."
        self._finished.set()

class DataChannel(metaclass=abc.ABCMeta):
    """Represents a channel for reading and writing data over the data plane.

  Read data and timer from this channel with the input_elements method::

    for elements_data in data_channel.input_elements(
        instruction_id, transform_ids, timers):
      [process elements_data]

  Write data to this channel using the output_stream method::

    out1 = data_channel.output_stream(instruction_id, transform_id)
    out1.write(...)
    out1.close()

  Write timer to this channel using the output_timer_stream method::

    out1 = data_channel.output_timer_stream(instruction_id,
                                            transform_id,
                                            timer_family_id)
    out1.write(...)
    out1.close()

  When all data/timer for all instructions is written, close the channel::

    data_channel.close()
  """

    @abc.abstractmethod
    def input_elements(self, instruction_id, expected_inputs, abort_callback=None):
        if False:
            print('Hello World!')
        'Returns an iterable of all Element.Data and Element.Timers bundles for\n    instruction_id.\n\n    This iterable terminates only once the full set of data has been recieved\n    for each of the expected transforms. It may block waiting for more data.\n\n    Args:\n        instruction_id: which instruction the results must belong to\n        expected_inputs: which transforms to wait on for completion\n        abort_callback: a callback to invoke if blocking returning whether\n            to abort before consuming all the data\n    '
        raise NotImplementedError(type(self))

    @abc.abstractmethod
    def output_stream(self, instruction_id, transform_id):
        if False:
            while True:
                i = 10
        'Returns an output stream writing elements to transform_id.\n\n    Args:\n        instruction_id: which instruction this stream belongs to\n        transform_id: the transform_id of the returned stream\n    '
        raise NotImplementedError(type(self))

    @abc.abstractmethod
    def output_timer_stream(self, instruction_id, transform_id, timer_family_id):
        if False:
            i = 10
            return i + 15
        'Returns an output stream written timers to transform_id.\n\n    Args:\n        instruction_id: which instruction this stream belongs to\n        transform_id: the transform_id of the returned stream\n        timer_family_id: the timer family of the written timer\n    '
        raise NotImplementedError(type(self))

    @abc.abstractmethod
    def close(self):
        if False:
            return 10
        'Closes this channel, indicating that all data has been written.\n\n    Data can continue to be read.\n\n    If this channel is shared by many instructions, should only be called on\n    worker shutdown.\n    '
        raise NotImplementedError(type(self))

class InMemoryDataChannel(DataChannel):
    """An in-memory implementation of a DataChannel.

  This channel is two-sided.  What is written to one side is read by the other.
  The inverse() method returns the other side of a instance.
  """

    def __init__(self, inverse=None, data_buffer_time_limit_ms=0):
        if False:
            print('Hello World!')
        self._inputs = []
        self._data_buffer_time_limit_ms = data_buffer_time_limit_ms
        self._inverse = inverse or InMemoryDataChannel(self, data_buffer_time_limit_ms=data_buffer_time_limit_ms)

    def inverse(self):
        if False:
            print('Hello World!')
        return self._inverse

    def input_elements(self, instruction_id, unused_expected_inputs, abort_callback=None):
        if False:
            while True:
                i = 10
        other_inputs = []
        for element in self._inputs:
            if element.instruction_id == instruction_id:
                if isinstance(element, beam_fn_api_pb2.Elements.Timers):
                    if not element.is_last:
                        yield element
                if isinstance(element, beam_fn_api_pb2.Elements.Data):
                    if element.data or element.is_last:
                        yield element
            else:
                other_inputs.append(element)
        self._inputs = other_inputs

    def output_timer_stream(self, instruction_id, transform_id, timer_family_id):
        if False:
            while True:
                i = 10

        def add_to_inverse_output(timer):
            if False:
                return 10
            if timer:
                self._inverse._inputs.append(beam_fn_api_pb2.Elements.Timers(instruction_id=instruction_id, transform_id=transform_id, timer_family_id=timer_family_id, timers=timer, is_last=False))

        def close_stream(timer):
            if False:
                i = 10
                return i + 15
            add_to_inverse_output(timer)
            self._inverse._inputs.append(beam_fn_api_pb2.Elements.Timers(instruction_id=instruction_id, transform_id=transform_id, timer_family_id='', is_last=True))
        return ClosableOutputStream.create(add_to_inverse_output, close_stream, self._data_buffer_time_limit_ms)

    def output_stream(self, instruction_id, transform_id):
        if False:
            print('Hello World!')

        def add_to_inverse_output(data):
            if False:
                return 10
            self._inverse._inputs.append(beam_fn_api_pb2.Elements.Data(instruction_id=instruction_id, transform_id=transform_id, data=data))
        return ClosableOutputStream.create(add_to_inverse_output, add_to_inverse_output, self._data_buffer_time_limit_ms)

    def close(self):
        if False:
            while True:
                i = 10
        pass

class _GrpcDataChannel(DataChannel):
    """Base class for implementing a BeamFnData-based DataChannel."""
    _WRITES_FINISHED = object()

    def __init__(self, data_buffer_time_limit_ms=0):
        if False:
            for i in range(10):
                print('nop')
        self._data_buffer_time_limit_ms = data_buffer_time_limit_ms
        self._to_send = queue.Queue()
        self._received = collections.defaultdict(lambda : queue.Queue(maxsize=5))
        self._cleaned_instruction_ids = collections.OrderedDict()
        self._receive_lock = threading.Lock()
        self._reads_finished = threading.Event()
        self._closed = False
        self._exception = None

    def close(self):
        if False:
            i = 10
            return i + 15
        self._to_send.put(self._WRITES_FINISHED)
        self._closed = True

    def wait(self, timeout=None):
        if False:
            print('Hello World!')
        self._reads_finished.wait(timeout)

    def _receiving_queue(self, instruction_id):
        if False:
            print('Hello World!')
        '\n    Gets or creates queue for a instruction_id. Or, returns None if the\n    instruction_id is already cleaned up. This is best-effort as we track\n    a limited number of cleaned-up instructions.\n    '
        with self._receive_lock:
            if instruction_id in self._cleaned_instruction_ids:
                return None
            return self._received[instruction_id]

    def _clean_receiving_queue(self, instruction_id):
        if False:
            print('Hello World!')
        '\n    Removes the queue and adds the instruction_id to the cleaned-up list. The\n    instruction_id cannot be reused for new queue.\n    '
        with self._receive_lock:
            self._received.pop(instruction_id)
            self._cleaned_instruction_ids[instruction_id] = True
            while len(self._cleaned_instruction_ids) > _MAX_CLEANED_INSTRUCTIONS:
                self._cleaned_instruction_ids.popitem(last=False)

    def input_elements(self, instruction_id, expected_inputs, abort_callback=None):
        if False:
            i = 10
            return i + 15
        '\n    Generator to retrieve elements for an instruction_id\n    input_elements should be called only once for an instruction_id\n\n    Args:\n      instruction_id(str): instruction_id for which data is read\n      expected_inputs(collection): expected inputs, include both data and timer.\n    '
        received = self._receiving_queue(instruction_id)
        if received is None:
            raise RuntimeError('Instruction cleaned up already %s' % instruction_id)
        done_inputs = set()
        abort_callback = abort_callback or (lambda : False)
        log_interval_sec = 5 * 60
        try:
            start_time = time.time()
            next_waiting_log_time = start_time + log_interval_sec
            while len(done_inputs) < len(expected_inputs):
                try:
                    element = received.get(timeout=1)
                except queue.Empty:
                    if self._closed:
                        raise RuntimeError('Channel closed prematurely.')
                    if abort_callback():
                        return
                    if self._exception:
                        raise self._exception from None
                    current_time = time.time()
                    if next_waiting_log_time <= current_time:
                        _LOGGER.info('Detected input queue delay longer than %s seconds. Waiting to receive elements in input queue for instruction: %s for %.2f seconds.', log_interval_sec, instruction_id, current_time - start_time)
                        next_waiting_log_time = current_time + log_interval_sec
                else:
                    start_time = time.time()
                    next_waiting_log_time = start_time + log_interval_sec
                    if isinstance(element, beam_fn_api_pb2.Elements.Timers):
                        if element.is_last:
                            done_inputs.add((element.transform_id, element.timer_family_id))
                        else:
                            yield element
                    elif isinstance(element, beam_fn_api_pb2.Elements.Data):
                        if element.is_last:
                            done_inputs.add(element.transform_id)
                        else:
                            assert element.transform_id not in done_inputs
                            yield element
                    else:
                        raise ValueError('Unexpected input element type %s' % type(element))
        finally:
            self._clean_receiving_queue(instruction_id)

    def output_stream(self, instruction_id, transform_id):
        if False:
            i = 10
            return i + 15

        def add_to_send_queue(data):
            if False:
                return 10
            if data:
                self._to_send.put(beam_fn_api_pb2.Elements.Data(instruction_id=instruction_id, transform_id=transform_id, data=data))

        def close_callback(data):
            if False:
                return 10
            add_to_send_queue(data)
            self._to_send.put(beam_fn_api_pb2.Elements.Data(instruction_id=instruction_id, transform_id=transform_id, is_last=True))
        return ClosableOutputStream.create(close_callback, add_to_send_queue, self._data_buffer_time_limit_ms)

    def output_timer_stream(self, instruction_id, transform_id, timer_family_id):
        if False:
            i = 10
            return i + 15

        def add_to_send_queue(timer):
            if False:
                for i in range(10):
                    print('nop')
            if timer:
                self._to_send.put(beam_fn_api_pb2.Elements.Timers(instruction_id=instruction_id, transform_id=transform_id, timer_family_id=timer_family_id, timers=timer, is_last=False))

        def close_callback(timer):
            if False:
                i = 10
                return i + 15
            add_to_send_queue(timer)
            self._to_send.put(beam_fn_api_pb2.Elements.Timers(instruction_id=instruction_id, transform_id=transform_id, timer_family_id=timer_family_id, is_last=True))
        return ClosableOutputStream.create(close_callback, add_to_send_queue, self._data_buffer_time_limit_ms)

    def _write_outputs(self):
        if False:
            i = 10
            return i + 15
        stream_done = False
        while not stream_done:
            streams = [self._to_send.get()]
            try:
                for _ in range(100):
                    streams.append(self._to_send.get_nowait())
            except queue.Empty:
                pass
            if streams[-1] is self._WRITES_FINISHED:
                stream_done = True
                streams.pop()
            if streams:
                data_stream = []
                timer_stream = []
                for stream in streams:
                    if isinstance(stream, beam_fn_api_pb2.Elements.Timers):
                        timer_stream.append(stream)
                    elif isinstance(stream, beam_fn_api_pb2.Elements.Data):
                        data_stream.append(stream)
                    else:
                        raise ValueError('Unexpected output element type %s' % type(stream))
                yield beam_fn_api_pb2.Elements(data=data_stream, timers=timer_stream)

    def _read_inputs(self, elements_iterator):
        if False:
            for i in range(10):
                print('nop')
        next_discard_log_time = 0

        def _put_queue(instruction_id, element):
            if False:
                for i in range(10):
                    print('nop')
            '\n      Puts element to the queue of the instruction_id, or discards it if the\n      instruction_id is already cleaned up.\n      '
            nonlocal next_discard_log_time
            start_time = time.time()
            next_waiting_log_time = start_time + 300
            while True:
                input_queue = self._receiving_queue(instruction_id)
                if input_queue is None:
                    current_time = time.time()
                    if next_discard_log_time <= current_time:
                        _LOGGER.info('Discard inputs for cleaned up instruction: %s', instruction_id)
                        next_discard_log_time = current_time + 10
                    return
                try:
                    input_queue.put(element, timeout=1)
                    return
                except queue.Full:
                    current_time = time.time()
                    if next_waiting_log_time <= current_time:
                        _LOGGER.info('Waiting on input queue of instruction: %s for %.2f seconds', instruction_id, current_time - start_time)
                        next_waiting_log_time = current_time + 300
        try:
            for elements in elements_iterator:
                for timer in elements.timers:
                    _put_queue(timer.instruction_id, timer)
                for data in elements.data:
                    _put_queue(data.instruction_id, data)
        except Exception as e:
            if not self._closed:
                _LOGGER.exception('Failed to read inputs in the data plane.')
                self._exception = e
                raise
        finally:
            self._closed = True
            self._reads_finished.set()

    def set_inputs(self, elements_iterator):
        if False:
            print('Hello World!')
        reader = threading.Thread(target=lambda : self._read_inputs(elements_iterator), name='read_grpc_client_inputs')
        reader.daemon = True
        reader.start()

class GrpcClientDataChannel(_GrpcDataChannel):
    """A DataChannel wrapping the client side of a BeamFnData connection."""

    def __init__(self, data_stub, data_buffer_time_limit_ms=0):
        if False:
            while True:
                i = 10
        super().__init__(data_buffer_time_limit_ms)
        self.set_inputs(data_stub.Data(self._write_outputs()))

class BeamFnDataServicer(beam_fn_api_pb2_grpc.BeamFnDataServicer):
    """Implementation of BeamFnDataServicer for any number of clients"""

    def __init__(self, data_buffer_time_limit_ms=0):
        if False:
            print('Hello World!')
        self._lock = threading.Lock()
        self._connections_by_worker_id = collections.defaultdict(lambda : _GrpcDataChannel(data_buffer_time_limit_ms))

    def get_conn_by_worker_id(self, worker_id):
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            return self._connections_by_worker_id[worker_id]

    def Data(self, elements_iterator, context):
        if False:
            return 10
        worker_id = dict(context.invocation_metadata())['worker_id']
        data_conn = self.get_conn_by_worker_id(worker_id)
        data_conn.set_inputs(elements_iterator)
        for elements in data_conn._write_outputs():
            yield elements

class DataChannelFactory(metaclass=abc.ABCMeta):
    """An abstract factory for creating ``DataChannel``."""

    @abc.abstractmethod
    def create_data_channel(self, remote_grpc_port):
        if False:
            while True:
                i = 10
        'Returns a ``DataChannel`` from the given RemoteGrpcPort.'
        raise NotImplementedError(type(self))

    @abc.abstractmethod
    def create_data_channel_from_url(self, url):
        if False:
            while True:
                i = 10
        'Returns a ``DataChannel`` from the given url.'
        raise NotImplementedError(type(self))

    @abc.abstractmethod
    def close(self):
        if False:
            print('Hello World!')
        'Close all channels that this factory owns.'
        raise NotImplementedError(type(self))

class GrpcClientDataChannelFactory(DataChannelFactory):
    """A factory for ``GrpcClientDataChannel``.

  Caches the created channels by ``data descriptor url``.
  """

    def __init__(self, credentials=None, worker_id=None, data_buffer_time_limit_ms=0):
        if False:
            i = 10
            return i + 15
        self._data_channel_cache = {}
        self._lock = threading.Lock()
        self._credentials = None
        self._worker_id = worker_id
        self._data_buffer_time_limit_ms = data_buffer_time_limit_ms
        if credentials is not None:
            _LOGGER.info('Using secure channel creds.')
            self._credentials = credentials

    def create_data_channel_from_url(self, url):
        if False:
            i = 10
            return i + 15
        if not url:
            return None
        if url not in self._data_channel_cache:
            with self._lock:
                if url not in self._data_channel_cache:
                    _LOGGER.info('Creating client data channel for %s', url)
                    channel_options = [('grpc.max_receive_message_length', -1), ('grpc.max_send_message_length', -1), ('grpc.service_config', _GRPC_SERVICE_CONFIG)]
                    grpc_channel = None
                    if self._credentials is None:
                        grpc_channel = GRPCChannelFactory.insecure_channel(url, options=channel_options)
                    else:
                        grpc_channel = GRPCChannelFactory.secure_channel(url, self._credentials, options=channel_options)
                    grpc_channel = grpc.intercept_channel(grpc_channel, WorkerIdInterceptor(self._worker_id))
                    self._data_channel_cache[url] = GrpcClientDataChannel(beam_fn_api_pb2_grpc.BeamFnDataStub(grpc_channel), self._data_buffer_time_limit_ms)
        return self._data_channel_cache[url]

    def create_data_channel(self, remote_grpc_port):
        if False:
            i = 10
            return i + 15
        url = remote_grpc_port.api_service_descriptor.url
        return self.create_data_channel_from_url(url)

    def close(self):
        if False:
            return 10
        _LOGGER.info('Closing all cached grpc data channels.')
        for (_, channel) in self._data_channel_cache.items():
            channel.close()
        self._data_channel_cache.clear()

class InMemoryDataChannelFactory(DataChannelFactory):
    """A singleton factory for ``InMemoryDataChannel``."""

    def __init__(self, in_memory_data_channel):
        if False:
            return 10
        self._in_memory_data_channel = in_memory_data_channel

    def create_data_channel(self, unused_remote_grpc_port):
        if False:
            i = 10
            return i + 15
        return self._in_memory_data_channel

    def create_data_channel_from_url(self, url):
        if False:
            for i in range(10):
                print('nop')
        return self._in_memory_data_channel

    def close(self):
        if False:
            while True:
                i = 10
        pass