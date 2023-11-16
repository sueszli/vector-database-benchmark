"""SDK harness for executing Python Fns via the Fn API."""
import base64
import bisect
import collections
import copy
import json
import logging
import random
import threading
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Container
from typing import DefaultDict
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from typing import cast
from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2
import apache_beam as beam
from apache_beam import coders
from apache_beam.coders import WindowedValueCoder
from apache_beam.coders import coder_impl
from apache_beam.internal import pickler
from apache_beam.io import iobase
from apache_beam.metrics import monitoring_infos
from apache_beam.portability import common_urns
from apache_beam.portability import python_urns
from apache_beam.portability.api import beam_fn_api_pb2
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.runners import common
from apache_beam.runners import pipeline_context
from apache_beam.runners.worker import data_sampler
from apache_beam.runners.worker import operation_specs
from apache_beam.runners.worker import operations
from apache_beam.runners.worker import statesampler
from apache_beam.transforms import TimeDomain
from apache_beam.transforms import core
from apache_beam.transforms import environments
from apache_beam.transforms import sideinputs
from apache_beam.transforms import userstate
from apache_beam.transforms import window
from apache_beam.utils import counters
from apache_beam.utils import proto_utils
from apache_beam.utils import timestamp
from apache_beam.utils.windowed_value import WindowedValue
if TYPE_CHECKING:
    from google.protobuf import message
    from apache_beam import pvalue
    from apache_beam.portability.api import metrics_pb2
    from apache_beam.runners.sdf_utils import SplitResultPrimary
    from apache_beam.runners.sdf_utils import SplitResultResidual
    from apache_beam.runners.worker import data_plane
    from apache_beam.runners.worker import sdk_worker
    from apache_beam.transforms.core import Windowing
    from apache_beam.transforms.window import BoundedWindow
    from apache_beam.utils import windowed_value
T = TypeVar('T')
ConstructorFn = Callable[['BeamTransformFactory', Any, beam_runner_api_pb2.PTransform, Union['message.Message', bytes], Dict[str, List[operations.Operation]]], operations.Operation]
OperationT = TypeVar('OperationT', bound=operations.Operation)
FnApiUserRuntimeStateTypes = Union['ReadModifyWriteRuntimeState', 'CombiningValueRuntimeState', 'SynchronousSetRuntimeState', 'SynchronousBagRuntimeState']
DATA_INPUT_URN = 'beam:runner:source:v1'
DATA_OUTPUT_URN = 'beam:runner:sink:v1'
SYNTHETIC_DATA_SAMPLING_URN = 'beam:internal:sampling:v1'
IDENTITY_DOFN_URN = 'beam:dofn:identity:0.1'
OLD_DATAFLOW_RUNNER_HARNESS_PARDO_URN = 'beam:dofn:javasdk:0.1'
OLD_DATAFLOW_RUNNER_HARNESS_READ_URN = 'beam:source:java:0.1'
URNS_NEEDING_PCOLLECTIONS = set([monitoring_infos.ELEMENT_COUNT_URN, monitoring_infos.SAMPLED_BYTE_SIZE_URN])
_LOGGER = logging.getLogger(__name__)

class RunnerIOOperation(operations.Operation):
    """Common baseclass for runner harness IO operations."""

    def __init__(self, name_context, step_name, consumers, counter_factory, state_sampler, windowed_coder, transform_id, data_channel):
        if False:
            print('Hello World!')
        super().__init__(name_context, None, counter_factory, state_sampler)
        self.windowed_coder = windowed_coder
        self.windowed_coder_impl = windowed_coder.get_impl()
        self.transform_id = transform_id
        self.data_channel = data_channel
        for (_, consumer_ops) in consumers.items():
            for consumer in consumer_ops:
                self.add_receiver(consumer, 0)

class DataOutputOperation(RunnerIOOperation):
    """A sink-like operation that gathers outputs to be sent back to the runner.
  """

    def set_output_stream(self, output_stream):
        if False:
            i = 10
            return i + 15
        self.output_stream = output_stream

    def process(self, windowed_value):
        if False:
            while True:
                i = 10
        self.windowed_coder_impl.encode_to_stream(windowed_value, self.output_stream, True)
        self.output_stream.maybe_flush()

    def finish(self):
        if False:
            return 10
        super().finish()
        self.output_stream.close()

class DataInputOperation(RunnerIOOperation):
    """A source-like operation that gathers input from the runner."""

    def __init__(self, operation_name, step_name, consumers, counter_factory, state_sampler, windowed_coder, transform_id, data_channel):
        if False:
            while True:
                i = 10
        super().__init__(operation_name, step_name, consumers, counter_factory, state_sampler, windowed_coder, transform_id=transform_id, data_channel=data_channel)
        self.consumer = next(iter(consumers.values()))
        self.splitting_lock = threading.Lock()
        self.index = -1
        self.stop = float('inf')
        self.started = False

    def setup(self, data_sampler=None):
        if False:
            print('Hello World!')
        super().setup(data_sampler)
        self.receivers = [operations.ConsumerSet.create(counter_factory=self.counter_factory, step_name=self.name_context.step_name, output_index=0, consumers=self.consumer, coder=self.windowed_coder, producer_type_hints=self._get_runtime_performance_hints(), producer_batch_converter=self.get_output_batch_converter())]

    def start(self):
        if False:
            return 10
        super().start()
        with self.splitting_lock:
            self.started = True

    def process(self, windowed_value):
        if False:
            i = 10
            return i + 15
        self.output(windowed_value)

    def process_encoded(self, encoded_windowed_values):
        if False:
            i = 10
            return i + 15
        input_stream = coder_impl.create_InputStream(encoded_windowed_values)
        while input_stream.size() > 0:
            with self.splitting_lock:
                if self.index == self.stop - 1:
                    return
                self.index += 1
            try:
                decoded_value = self.windowed_coder_impl.decode_from_stream(input_stream, True)
            except Exception as exn:
                raise ValueError('Error decoding input stream with coder ' + str(self.windowed_coder)) from exn
            self.output(decoded_value)

    def monitoring_infos(self, transform_id, tag_to_pcollection_id):
        if False:
            i = 10
            return i + 15
        all_monitoring_infos = super().monitoring_infos(transform_id, tag_to_pcollection_id)
        read_progress_info = monitoring_infos.int64_counter(monitoring_infos.DATA_CHANNEL_READ_INDEX, self.index, ptransform=transform_id)
        all_monitoring_infos[monitoring_infos.to_key(read_progress_info)] = read_progress_info
        return all_monitoring_infos

    def try_split(self, fraction_of_remainder, total_buffer_size, allowed_split_points):
        if False:
            return 10
        with self.splitting_lock:
            if not self.started:
                return None
            if self.index == -1:
                current_element_progress = 1.0
            else:
                current_element_progress_object = self.receivers[0].current_element_progress()
                if current_element_progress_object is None:
                    current_element_progress = 0.5
                else:
                    current_element_progress = current_element_progress_object.fraction_completed
            split = self._compute_split(self.index, current_element_progress, self.stop, fraction_of_remainder, total_buffer_size, allowed_split_points, self.receivers[0].try_split)
            if split:
                self.stop = split[-1]
            return split

    @staticmethod
    def _compute_split(index, current_element_progress, stop, fraction_of_remainder, total_buffer_size, allowed_split_points=(), try_split=lambda fraction: None):
        if False:
            return 10

        def is_valid_split_point(index):
            if False:
                return 10
            return not allowed_split_points or index in allowed_split_points
        if total_buffer_size < index + 1:
            total_buffer_size = index + 1
        elif total_buffer_size > stop:
            total_buffer_size = stop
        remainder = total_buffer_size - index - current_element_progress
        keep = remainder * fraction_of_remainder
        if current_element_progress < 1:
            keep_of_element_remainder = keep / (1 - current_element_progress)
            if keep_of_element_remainder < 1 and is_valid_split_point(index) and is_valid_split_point(index + 1):
                split = try_split(keep_of_element_remainder)
                if split:
                    (element_primaries, element_residuals) = split
                    return (index - 1, element_primaries, element_residuals, index + 1)
        stop_index = index + max(1, int(round(current_element_progress + keep)))
        if allowed_split_points and stop_index not in allowed_split_points:
            allowed_split_points = sorted(allowed_split_points)
            closest = bisect.bisect(allowed_split_points, stop_index)
            if closest == 0:
                stop_index = allowed_split_points[0]
            elif closest == len(allowed_split_points):
                stop_index = allowed_split_points[-1]
            else:
                prev = allowed_split_points[closest - 1]
                next = allowed_split_points[closest]
                if index < prev and stop_index - prev < next - stop_index:
                    stop_index = prev
                else:
                    stop_index = next
        if index < stop_index < stop:
            return (stop_index - 1, [], [], stop_index)
        else:
            return None

    def finish(self):
        if False:
            i = 10
            return i + 15
        super().finish()
        with self.splitting_lock:
            self.index += 1
            self.started = False

    def reset(self):
        if False:
            return 10
        with self.splitting_lock:
            self.index = -1
            self.stop = float('inf')
        super().reset()

class _StateBackedIterable(object):

    def __init__(self, state_handler, state_key, coder_or_impl):
        if False:
            print('Hello World!')
        self._state_handler = state_handler
        self._state_key = state_key
        if isinstance(coder_or_impl, coders.Coder):
            self._coder_impl = coder_or_impl.get_impl()
        else:
            self._coder_impl = coder_or_impl

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self._state_handler.blocking_get(self._state_key, self._coder_impl))

    def __reduce__(self):
        if False:
            print('Hello World!')
        return (list, (list(self),))
coder_impl.FastPrimitivesCoderImpl.register_iterable_like_type(_StateBackedIterable)

class StateBackedSideInputMap(object):

    def __init__(self, state_handler, transform_id, tag, side_input_data, coder):
        if False:
            for i in range(10):
                print('nop')
        self._state_handler = state_handler
        self._transform_id = transform_id
        self._tag = tag
        self._side_input_data = side_input_data
        self._element_coder = coder.wrapped_value_coder
        self._target_window_coder = coder.window_coder
        self._cache = {}

    def __getitem__(self, window):
        if False:
            while True:
                i = 10
        target_window = self._side_input_data.window_mapping_fn(window)
        if target_window not in self._cache:
            state_handler = self._state_handler
            access_pattern = self._side_input_data.access_pattern
            if access_pattern == common_urns.side_inputs.ITERABLE.urn:
                state_key = beam_fn_api_pb2.StateKey(iterable_side_input=beam_fn_api_pb2.StateKey.IterableSideInput(transform_id=self._transform_id, side_input_id=self._tag, window=self._target_window_coder.encode(target_window)))
                raw_view = _StateBackedIterable(state_handler, state_key, self._element_coder)
            elif access_pattern == common_urns.side_inputs.MULTIMAP.urn:
                state_key = beam_fn_api_pb2.StateKey(multimap_side_input=beam_fn_api_pb2.StateKey.MultimapSideInput(transform_id=self._transform_id, side_input_id=self._tag, window=self._target_window_coder.encode(target_window), key=b''))
                cache = {}
                key_coder_impl = self._element_coder.key_coder().get_impl()
                value_coder = self._element_coder.value_coder()

                class MultiMap(object):

                    def __getitem__(self, key):
                        if False:
                            return 10
                        if key not in cache:
                            keyed_state_key = beam_fn_api_pb2.StateKey()
                            keyed_state_key.CopyFrom(state_key)
                            keyed_state_key.multimap_side_input.key = key_coder_impl.encode_nested(key)
                            cache[key] = _StateBackedIterable(state_handler, keyed_state_key, value_coder)
                        return cache[key]

                    def __reduce__(self):
                        if False:
                            while True:
                                i = 10
                        raise TypeError(common_urns.side_inputs.MULTIMAP.urn)
                raw_view = MultiMap()
            else:
                raise ValueError("Unknown access pattern: '%s'" % access_pattern)
            self._cache[target_window] = self._side_input_data.view_fn(raw_view)
        return self._cache[target_window]

    def is_globally_windowed(self):
        if False:
            for i in range(10):
                print('nop')
        return self._side_input_data.window_mapping_fn == sideinputs._global_window_mapping_fn

    def reset(self):
        if False:
            i = 10
            return i + 15
        self._cache = {}

class ReadModifyWriteRuntimeState(userstate.ReadModifyWriteRuntimeState):

    def __init__(self, underlying_bag_state):
        if False:
            while True:
                i = 10
        self._underlying_bag_state = underlying_bag_state

    def read(self):
        if False:
            i = 10
            return i + 15
        values = list(self._underlying_bag_state.read())
        if not values:
            return None
        return values[0]

    def write(self, value):
        if False:
            while True:
                i = 10
        self.clear()
        self._underlying_bag_state.add(value)

    def clear(self):
        if False:
            i = 10
            return i + 15
        self._underlying_bag_state.clear()

    def commit(self):
        if False:
            for i in range(10):
                print('nop')
        self._underlying_bag_state.commit()

class CombiningValueRuntimeState(userstate.CombiningValueRuntimeState):

    def __init__(self, underlying_bag_state, combinefn):
        if False:
            for i in range(10):
                print('nop')
        self._combinefn = combinefn
        self._combinefn.setup()
        self._underlying_bag_state = underlying_bag_state
        self._finalized = False

    def _read_accumulator(self, rewrite=True):
        if False:
            while True:
                i = 10
        merged_accumulator = self._combinefn.merge_accumulators(self._underlying_bag_state.read())
        if rewrite:
            self._underlying_bag_state.clear()
            self._underlying_bag_state.add(merged_accumulator)
        return merged_accumulator

    def read(self):
        if False:
            print('Hello World!')
        return self._combinefn.extract_output(self._read_accumulator())

    def add(self, value):
        if False:
            for i in range(10):
                print('nop')
        if random.random() < 0.5:
            accumulator = self._read_accumulator(False)
            self._underlying_bag_state.clear()
        else:
            accumulator = self._combinefn.create_accumulator()
        self._underlying_bag_state.add(self._combinefn.add_input(accumulator, value))

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        self._underlying_bag_state.clear()

    def commit(self):
        if False:
            while True:
                i = 10
        self._underlying_bag_state.commit()

    def finalize(self):
        if False:
            while True:
                i = 10
        if not self._finalized:
            self._combinefn.teardown()
            self._finalized = True

class _ConcatIterable(object):
    """An iterable that is the concatination of two iterables.

  Unlike itertools.chain, this allows reiteration.
  """

    def __init__(self, first, second):
        if False:
            return 10
        self.first = first
        self.second = second

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        for elem in self.first:
            yield elem
        for elem in self.second:
            yield elem
coder_impl.FastPrimitivesCoderImpl.register_iterable_like_type(_ConcatIterable)

class SynchronousBagRuntimeState(userstate.BagRuntimeState):

    def __init__(self, state_handler, state_key, value_coder):
        if False:
            i = 10
            return i + 15
        self._state_handler = state_handler
        self._state_key = state_key
        self._value_coder = value_coder
        self._cleared = False
        self._added_elements = []

    def read(self):
        if False:
            return 10
        return _ConcatIterable([] if self._cleared else cast('Iterable[Any]', _StateBackedIterable(self._state_handler, self._state_key, self._value_coder)), self._added_elements)

    def add(self, value):
        if False:
            return 10
        self._added_elements.append(value)

    def clear(self):
        if False:
            print('Hello World!')
        self._cleared = True
        self._added_elements = []

    def commit(self):
        if False:
            for i in range(10):
                print('nop')
        to_await = None
        if self._cleared:
            to_await = self._state_handler.clear(self._state_key)
        if self._added_elements:
            to_await = self._state_handler.extend(self._state_key, self._value_coder.get_impl(), self._added_elements)
        if to_await:
            to_await.get()

class SynchronousSetRuntimeState(userstate.SetRuntimeState):

    def __init__(self, state_handler, state_key, value_coder):
        if False:
            for i in range(10):
                print('nop')
        self._state_handler = state_handler
        self._state_key = state_key
        self._value_coder = value_coder
        self._cleared = False
        self._added_elements = set()

    def _compact_data(self, rewrite=True):
        if False:
            while True:
                i = 10
        accumulator = set(_ConcatIterable(set() if self._cleared else _StateBackedIterable(self._state_handler, self._state_key, self._value_coder), self._added_elements))
        if rewrite and accumulator:
            self._state_handler.clear(self._state_key)
            self._state_handler.extend(self._state_key, self._value_coder.get_impl(), accumulator)
            self._added_elements = set()
        return accumulator

    def read(self):
        if False:
            return 10
        return self._compact_data(rewrite=False)

    def add(self, value):
        if False:
            print('Hello World!')
        if self._cleared:
            self._state_handler.clear(self._state_key)
            self._cleared = False
        self._added_elements.add(value)
        if random.random() > 0.5:
            self._compact_data()

    def clear(self):
        if False:
            i = 10
            return i + 15
        self._cleared = True
        self._added_elements = set()

    def commit(self):
        if False:
            return 10
        to_await = None
        if self._cleared:
            to_await = self._state_handler.clear(self._state_key)
        if self._added_elements:
            to_await = self._state_handler.extend(self._state_key, self._value_coder.get_impl(), self._added_elements)
        if to_await:
            to_await.get()

class OutputTimer(userstate.BaseTimer):

    def __init__(self, key, window, timestamp, paneinfo, time_domain, timer_family_id, timer_coder_impl, output_stream):
        if False:
            for i in range(10):
                print('nop')
        self._key = key
        self._window = window
        self._input_timestamp = timestamp
        self._paneinfo = paneinfo
        self._time_domain = time_domain
        self._timer_family_id = timer_family_id
        self._output_stream = output_stream
        self._timer_coder_impl = timer_coder_impl

    def set(self, ts: timestamp.TimestampTypes, dynamic_timer_tag='') -> None:
        if False:
            for i in range(10):
                print('nop')
        ts = timestamp.Timestamp.of(ts)
        timer = userstate.Timer(user_key=self._key, dynamic_timer_tag=dynamic_timer_tag, windows=(self._window,), clear_bit=False, fire_timestamp=ts, hold_timestamp=ts if TimeDomain.is_event_time(self._time_domain) else self._input_timestamp, paneinfo=self._paneinfo)
        self._timer_coder_impl.encode_to_stream(timer, self._output_stream, True)
        self._output_stream.maybe_flush()

    def clear(self, dynamic_timer_tag='') -> None:
        if False:
            print('Hello World!')
        timer = userstate.Timer(user_key=self._key, dynamic_timer_tag=dynamic_timer_tag, windows=(self._window,), clear_bit=True, fire_timestamp=None, hold_timestamp=None, paneinfo=None)
        self._timer_coder_impl.encode_to_stream(timer, self._output_stream, True)
        self._output_stream.maybe_flush()

class TimerInfo(object):
    """A data class to store information related to a timer."""

    def __init__(self, timer_coder_impl, output_stream=None):
        if False:
            while True:
                i = 10
        self.timer_coder_impl = timer_coder_impl
        self.output_stream = output_stream

class FnApiUserStateContext(userstate.UserStateContext):
    """Interface for state and timers from SDK to Fn API servicer of state.."""

    def __init__(self, state_handler, transform_id, key_coder, window_coder):
        if False:
            i = 10
            return i + 15
        'Initialize a ``FnApiUserStateContext``.\n\n    Args:\n      state_handler: A StateServicer object.\n      transform_id: The name of the PTransform that this context is associated.\n      key_coder: Coder for the key type.\n      window_coder: Coder for the window type.\n    '
        self._state_handler = state_handler
        self._transform_id = transform_id
        self._key_coder = key_coder
        self._window_coder = window_coder
        self._timers_info = {}
        self._all_states = {}

    def add_timer_info(self, timer_family_id, timer_info):
        if False:
            i = 10
            return i + 15
        self._timers_info[timer_family_id] = timer_info

    def get_timer(self, timer_spec: userstate.TimerSpec, key, window, timestamp, pane) -> OutputTimer:
        if False:
            while True:
                i = 10
        assert self._timers_info[timer_spec.name].output_stream is not None
        timer_coder_impl = self._timers_info[timer_spec.name].timer_coder_impl
        output_stream = self._timers_info[timer_spec.name].output_stream
        return OutputTimer(key, window, timestamp, pane, timer_spec.time_domain, timer_spec.name, timer_coder_impl, output_stream)

    def get_state(self, *args):
        if False:
            for i in range(10):
                print('nop')
        state_handle = self._all_states.get(args)
        if state_handle is None:
            state_handle = self._all_states[args] = self._create_state(*args)
        return state_handle

    def _create_state(self, state_spec, key, window):
        if False:
            while True:
                i = 10
        if isinstance(state_spec, (userstate.BagStateSpec, userstate.CombiningValueStateSpec, userstate.ReadModifyWriteStateSpec)):
            bag_state = SynchronousBagRuntimeState(self._state_handler, state_key=beam_fn_api_pb2.StateKey(bag_user_state=beam_fn_api_pb2.StateKey.BagUserState(transform_id=self._transform_id, user_state_id=state_spec.name, window=self._window_coder.encode(window), key=self._key_coder.encode_nested(key))), value_coder=state_spec.coder)
            if isinstance(state_spec, userstate.BagStateSpec):
                return bag_state
            elif isinstance(state_spec, userstate.ReadModifyWriteStateSpec):
                return ReadModifyWriteRuntimeState(bag_state)
            else:
                return CombiningValueRuntimeState(bag_state, copy.deepcopy(state_spec.combine_fn))
        elif isinstance(state_spec, userstate.SetStateSpec):
            return SynchronousSetRuntimeState(self._state_handler, state_key=beam_fn_api_pb2.StateKey(bag_user_state=beam_fn_api_pb2.StateKey.BagUserState(transform_id=self._transform_id, user_state_id=state_spec.name, window=self._window_coder.encode(window), key=self._key_coder.encode_nested(key))), value_coder=state_spec.coder)
        else:
            raise NotImplementedError(state_spec)

    def commit(self):
        if False:
            i = 10
            return i + 15
        for state in self._all_states.values():
            state.commit()

    def reset(self):
        if False:
            i = 10
            return i + 15
        for state in self._all_states.values():
            state.finalize()
        self._all_states = {}

def memoize(func):
    if False:
        print('Hello World!')
    cache = {}
    missing = object()

    def wrapper(*args):
        if False:
            return 10
        result = cache.get(args, missing)
        if result is missing:
            result = cache[args] = func(*args)
        return result
    return wrapper

def only_element(iterable):
    if False:
        return 10
    (element,) = iterable
    return element

def _environments_compatible(submission, runtime):
    if False:
        return 10
    if submission == runtime:
        return True
    if 'rc' in submission and runtime in submission:
        return True
    return False

def _verify_descriptor_created_in_a_compatible_env(process_bundle_descriptor):
    if False:
        i = 10
        return i + 15
    runtime_sdk = environments.sdk_base_version_capability()
    for t in process_bundle_descriptor.transforms.values():
        env = process_bundle_descriptor.environments[t.environment_id]
        for c in env.capabilities:
            if c.startswith(environments.SDK_VERSION_CAPABILITY_PREFIX) and (not _environments_compatible(c, runtime_sdk)):
                raise RuntimeError(f'Pipeline construction environment and pipeline runtime environment are not compatible. If you use a custom container image, check that the Python interpreter minor version and the Apache Beam version in your image match the versions used at pipeline construction time. Submission environment: {c}. Runtime environment: {runtime_sdk}.')

class BundleProcessor(object):
    """ A class for processing bundles of elements. """

    def __init__(self, process_bundle_descriptor, state_handler, data_channel_factory, data_sampler=None):
        if False:
            return 10
        'Initialize a bundle processor.\n\n    Args:\n      process_bundle_descriptor (``beam_fn_api_pb2.ProcessBundleDescriptor``):\n        a description of the stage that this ``BundleProcessor``is to execute.\n      state_handler (CachingStateHandler).\n      data_channel_factory (``data_plane.DataChannelFactory``).\n    '
        self.process_bundle_descriptor = process_bundle_descriptor
        self.state_handler = state_handler
        self.data_channel_factory = data_channel_factory
        self.data_sampler = data_sampler
        self.current_instruction_id = None
        _verify_descriptor_created_in_a_compatible_env(process_bundle_descriptor)
        if self.process_bundle_descriptor.timer_api_service_descriptor.url:
            self.timer_data_channel = data_channel_factory.create_data_channel_from_url(self.process_bundle_descriptor.timer_api_service_descriptor.url)
        else:
            self.timer_data_channel = None
        self.timers_info = {}
        self.counter_factory = counters.CounterFactory()
        self.state_sampler = statesampler.StateSampler('fnapi-step-%s' % self.process_bundle_descriptor.id, self.counter_factory)
        self.ops = self.create_execution_tree(self.process_bundle_descriptor)
        for op in reversed(self.ops.values()):
            op.setup(self.data_sampler)
        self.splitting_lock = threading.Lock()

    def create_execution_tree(self, descriptor):
        if False:
            return 10
        transform_factory = BeamTransformFactory(descriptor, self.data_channel_factory, self.counter_factory, self.state_sampler, self.state_handler, self.data_sampler)
        self.timers_info = transform_factory.extract_timers_info()

        def is_side_input(transform_proto, tag):
            if False:
                for i in range(10):
                    print('nop')
            if transform_proto.spec.urn == common_urns.primitives.PAR_DO.urn:
                return tag in proto_utils.parse_Bytes(transform_proto.spec.payload, beam_runner_api_pb2.ParDoPayload).side_inputs
        pcoll_consumers = collections.defaultdict(list)
        for (transform_id, transform_proto) in descriptor.transforms.items():
            for (tag, pcoll_id) in transform_proto.inputs.items():
                if not is_side_input(transform_proto, tag):
                    pcoll_consumers[pcoll_id].append(transform_id)

        @memoize
        def get_operation(transform_id):
            if False:
                for i in range(10):
                    print('nop')
            transform_consumers = {tag: [get_operation(op) for op in pcoll_consumers[pcoll_id]] for (tag, pcoll_id) in descriptor.transforms[transform_id].outputs.items()}
            if self.data_sampler:
                self.data_sampler.initialize_samplers(transform_id, descriptor, transform_factory.get_coder)
            return transform_factory.create_operation(transform_id, transform_consumers)

        @memoize
        def topological_height(transform_id):
            if False:
                return 10
            return 1 + max([0] + [topological_height(consumer) for pcoll in descriptor.transforms[transform_id].outputs.values() for consumer in pcoll_consumers[pcoll]])
        return collections.OrderedDict([(transform_id, cast(operations.DoOperation, get_operation(transform_id))) for transform_id in sorted(descriptor.transforms, key=topological_height, reverse=True)])

    def reset(self):
        if False:
            while True:
                i = 10
        self.counter_factory.reset()
        self.state_sampler.reset()
        for op in self.ops.values():
            op.reset()

    def process_bundle(self, instruction_id):
        if False:
            i = 10
            return i + 15
        expected_input_ops = []
        for op in self.ops.values():
            if isinstance(op, DataOutputOperation):
                op.set_output_stream(op.data_channel.output_stream(instruction_id, op.transform_id))
            elif isinstance(op, DataInputOperation):
                expected_input_ops.append(op)
        try:
            execution_context = ExecutionContext(instruction_id=instruction_id)
            self.current_instruction_id = instruction_id
            self.state_sampler.start()
            for op in reversed(self.ops.values()):
                _LOGGER.debug('start %s', op)
                op.execution_context = execution_context
                op.start()
            data_channels = collections.defaultdict(list)
            input_op_by_transform_id = {}
            for input_op in expected_input_ops:
                data_channels[input_op.data_channel].append(input_op.transform_id)
                input_op_by_transform_id[input_op.transform_id] = input_op
            if self.timer_data_channel:
                data_channels[self.timer_data_channel].extend(list(self.timers_info.keys()))
                for ((transform_id, timer_family_id), timer_info) in self.timers_info.items():
                    output_stream = self.timer_data_channel.output_timer_stream(instruction_id, transform_id, timer_family_id)
                    timer_info.output_stream = output_stream
                    self.ops[transform_id].add_timer_info(timer_family_id, timer_info)
            for (data_channel, expected_inputs) in data_channels.items():
                for element in data_channel.input_elements(instruction_id, expected_inputs):
                    if isinstance(element, beam_fn_api_pb2.Elements.Timers):
                        timer_coder_impl = self.timers_info[element.transform_id, element.timer_family_id].timer_coder_impl
                        for timer_data in timer_coder_impl.decode_all(element.timers):
                            self.ops[element.transform_id].process_timer(element.timer_family_id, timer_data)
                    elif isinstance(element, beam_fn_api_pb2.Elements.Data):
                        input_op_by_transform_id[element.transform_id].process_encoded(element.data)
            for op in self.ops.values():
                _LOGGER.debug('finish %s', op)
                op.finish()
            for timer_info in self.timers_info.values():
                assert timer_info.output_stream is not None
                timer_info.output_stream.close()
            return ([self.delayed_bundle_application(op, residual) for (op, residual) in execution_context.delayed_applications], self.requires_finalization())
        finally:
            with self.splitting_lock:
                self.current_instruction_id = None
            self.state_sampler.stop_if_still_running()

    def finalize_bundle(self):
        if False:
            i = 10
            return i + 15
        for op in self.ops.values():
            op.finalize_bundle()
        return beam_fn_api_pb2.FinalizeBundleResponse()

    def requires_finalization(self):
        if False:
            while True:
                i = 10
        return any((op.needs_finalization() for op in self.ops.values()))

    def try_split(self, bundle_split_request):
        if False:
            i = 10
            return i + 15
        split_response = beam_fn_api_pb2.ProcessBundleSplitResponse()
        with self.splitting_lock:
            if bundle_split_request.instruction_id != self.current_instruction_id:
                return split_response
            for op in self.ops.values():
                if isinstance(op, DataInputOperation):
                    desired_split = bundle_split_request.desired_splits.get(op.transform_id)
                    if desired_split:
                        split = op.try_split(desired_split.fraction_of_remainder, desired_split.estimated_input_elements, desired_split.allowed_split_points)
                        if split:
                            (primary_end, element_primaries, element_residuals, residual_start) = split
                            for element_primary in element_primaries:
                                split_response.primary_roots.add().CopyFrom(self.bundle_application(*element_primary))
                            for element_residual in element_residuals:
                                split_response.residual_roots.add().CopyFrom(self.delayed_bundle_application(*element_residual))
                            split_response.channel_splits.extend([beam_fn_api_pb2.ProcessBundleSplitResponse.ChannelSplit(transform_id=op.transform_id, last_primary_element=primary_end, first_residual_element=residual_start)])
        return split_response

    def delayed_bundle_application(self, op, deferred_remainder):
        if False:
            while True:
                i = 10
        assert op.input_info is not None
        (element_and_restriction, current_watermark, deferred_timestamp) = deferred_remainder
        if deferred_timestamp:
            assert isinstance(deferred_timestamp, timestamp.Duration)
            proto_deferred_watermark = proto_utils.from_micros(duration_pb2.Duration, deferred_timestamp.micros)
        else:
            proto_deferred_watermark = None
        return beam_fn_api_pb2.DelayedBundleApplication(requested_time_delay=proto_deferred_watermark, application=self.construct_bundle_application(op.input_info, current_watermark, element_and_restriction))

    def bundle_application(self, op, primary):
        if False:
            i = 10
            return i + 15
        assert op.input_info is not None
        return self.construct_bundle_application(op.input_info, None, primary.primary_value)

    def construct_bundle_application(self, op_input_info, output_watermark, element):
        if False:
            for i in range(10):
                print('nop')
        (transform_id, main_input_tag, main_input_coder, outputs) = op_input_info
        if output_watermark:
            proto_output_watermark = proto_utils.from_micros(timestamp_pb2.Timestamp, output_watermark.micros)
            output_watermarks = {output: proto_output_watermark for output in outputs}
        else:
            output_watermarks = None
        return beam_fn_api_pb2.BundleApplication(transform_id=transform_id, input_id=main_input_tag, output_watermarks=output_watermarks, element=main_input_coder.get_impl().encode_nested(element))

    def monitoring_infos(self):
        if False:
            i = 10
            return i + 15
        'Returns the list of MonitoringInfos collected processing this bundle.'
        all_monitoring_infos_dict = {}
        for (transform_id, op) in self.ops.items():
            tag_to_pcollection_id = self.process_bundle_descriptor.transforms[transform_id].outputs
            all_monitoring_infos_dict.update(op.monitoring_infos(transform_id, dict(tag_to_pcollection_id)))
        return list(all_monitoring_infos_dict.values())

    def shutdown(self):
        if False:
            return 10
        for op in self.ops.values():
            op.teardown()

@dataclass
class ExecutionContext:
    delayed_applications: List[Tuple[operations.DoOperation, common.SplitResultResidual]] = field(default_factory=list)
    output_sampler: Optional[data_sampler.OutputSampler] = None
    instruction_id: Optional[str] = None

class BeamTransformFactory(object):
    """Factory for turning transform_protos into executable operations."""

    def __init__(self, descriptor, data_channel_factory, counter_factory, state_sampler, state_handler, data_sampler):
        if False:
            i = 10
            return i + 15
        self.descriptor = descriptor
        self.data_channel_factory = data_channel_factory
        self.counter_factory = counter_factory
        self.state_sampler = state_sampler
        self.state_handler = state_handler
        self.context = pipeline_context.PipelineContext(descriptor, iterable_state_read=lambda token, element_coder_impl: _StateBackedIterable(state_handler, beam_fn_api_pb2.StateKey(runner=beam_fn_api_pb2.StateKey.Runner(key=token)), element_coder_impl))
        self.data_sampler = data_sampler
    _known_urns = {}

    @classmethod
    def register_urn(cls, urn, parameter_type):
        if False:
            print('Hello World!')

        def wrapper(func):
            if False:
                return 10
            cls._known_urns[urn] = (func, parameter_type)
            return func
        return wrapper

    def create_operation(self, transform_id, consumers):
        if False:
            for i in range(10):
                print('nop')
        transform_proto = self.descriptor.transforms[transform_id]
        if not transform_proto.unique_name:
            _LOGGER.debug('No unique name set for transform %s' % transform_id)
            transform_proto.unique_name = transform_id
        (creator, parameter_type) = self._known_urns[transform_proto.spec.urn]
        payload = proto_utils.parse_Bytes(transform_proto.spec.payload, parameter_type)
        return creator(self, transform_id, transform_proto, payload, consumers)

    def extract_timers_info(self):
        if False:
            for i in range(10):
                print('nop')
        timers_info = {}
        for (transform_id, transform_proto) in self.descriptor.transforms.items():
            if transform_proto.spec.urn == common_urns.primitives.PAR_DO.urn:
                pardo_payload = proto_utils.parse_Bytes(transform_proto.spec.payload, beam_runner_api_pb2.ParDoPayload)
                for (timer_family_id, timer_family_spec) in pardo_payload.timer_family_specs.items():
                    timer_coder_impl = self.get_coder(timer_family_spec.timer_family_coder_id).get_impl()
                    timers_info[transform_id, timer_family_id] = TimerInfo(timer_coder_impl=timer_coder_impl)
        return timers_info

    def get_coder(self, coder_id):
        if False:
            i = 10
            return i + 15
        if coder_id not in self.descriptor.coders:
            raise KeyError('No such coder: %s' % coder_id)
        coder_proto = self.descriptor.coders[coder_id]
        if coder_proto.spec.urn:
            return self.context.coders.get_by_id(coder_id)
        else:
            return operation_specs.get_coder_from_spec(json.loads(coder_proto.spec.payload.decode('utf-8')))

    def get_windowed_coder(self, pcoll_id):
        if False:
            print('Hello World!')
        coder = self.get_coder(self.descriptor.pcollections[pcoll_id].coder_id)
        if not isinstance(coder, WindowedValueCoder):
            windowing_strategy = self.descriptor.windowing_strategies[self.descriptor.pcollections[pcoll_id].windowing_strategy_id]
            return WindowedValueCoder(coder, self.get_coder(windowing_strategy.window_coder_id))
        else:
            return coder

    def get_output_coders(self, transform_proto):
        if False:
            i = 10
            return i + 15
        return {tag: self.get_windowed_coder(pcoll_id) for (tag, pcoll_id) in transform_proto.outputs.items()}

    def get_only_output_coder(self, transform_proto):
        if False:
            i = 10
            return i + 15
        return only_element(self.get_output_coders(transform_proto).values())

    def get_input_coders(self, transform_proto):
        if False:
            for i in range(10):
                print('nop')
        return {tag: self.get_windowed_coder(pcoll_id) for (tag, pcoll_id) in transform_proto.inputs.items()}

    def get_only_input_coder(self, transform_proto):
        if False:
            print('Hello World!')
        return only_element(list(self.get_input_coders(transform_proto).values()))

    def get_input_windowing(self, transform_proto):
        if False:
            while True:
                i = 10
        pcoll_id = only_element(transform_proto.inputs.values())
        windowing_strategy_id = self.descriptor.pcollections[pcoll_id].windowing_strategy_id
        return self.context.windowing_strategies.get_by_id(windowing_strategy_id)

    @staticmethod
    def augment_oldstyle_op(op, step_name, consumers, tag_list=None):
        if False:
            print('Hello World!')
        op.step_name = step_name
        for (tag, op_consumers) in consumers.items():
            for consumer in op_consumers:
                op.add_receiver(consumer, tag_list.index(tag) if tag_list else 0)
        return op

@BeamTransformFactory.register_urn(DATA_INPUT_URN, beam_fn_api_pb2.RemoteGrpcPort)
def create_source_runner(factory, transform_id, transform_proto, grpc_port, consumers):
    if False:
        i = 10
        return i + 15
    output_coder = factory.get_coder(grpc_port.coder_id)
    return DataInputOperation(common.NameContext(transform_proto.unique_name, transform_id), transform_proto.unique_name, consumers, factory.counter_factory, factory.state_sampler, output_coder, transform_id=transform_id, data_channel=factory.data_channel_factory.create_data_channel(grpc_port))

@BeamTransformFactory.register_urn(DATA_OUTPUT_URN, beam_fn_api_pb2.RemoteGrpcPort)
def create_sink_runner(factory, transform_id, transform_proto, grpc_port, consumers):
    if False:
        while True:
            i = 10
    output_coder = factory.get_coder(grpc_port.coder_id)
    return DataOutputOperation(common.NameContext(transform_proto.unique_name, transform_id), transform_proto.unique_name, consumers, factory.counter_factory, factory.state_sampler, output_coder, transform_id=transform_id, data_channel=factory.data_channel_factory.create_data_channel(grpc_port))

@BeamTransformFactory.register_urn(OLD_DATAFLOW_RUNNER_HARNESS_READ_URN, None)
def create_source_java(factory, transform_id, transform_proto, parameter, consumers):
    if False:
        while True:
            i = 10
    source = pickler.loads(base64.b64encode(parameter))
    spec = operation_specs.WorkerRead(iobase.SourceBundle(1.0, source, None, None), [factory.get_only_output_coder(transform_proto)])
    return factory.augment_oldstyle_op(operations.ReadOperation(common.NameContext(transform_proto.unique_name, transform_id), spec, factory.counter_factory, factory.state_sampler), transform_proto.unique_name, consumers)

@BeamTransformFactory.register_urn(common_urns.deprecated_primitives.READ.urn, beam_runner_api_pb2.ReadPayload)
def create_deprecated_read(factory, transform_id, transform_proto, parameter, consumers):
    if False:
        i = 10
        return i + 15
    source = iobase.BoundedSource.from_runner_api(parameter.source, factory.context)
    spec = operation_specs.WorkerRead(iobase.SourceBundle(1.0, source, None, None), [WindowedValueCoder(source.default_output_coder())])
    return factory.augment_oldstyle_op(operations.ReadOperation(common.NameContext(transform_proto.unique_name, transform_id), spec, factory.counter_factory, factory.state_sampler), transform_proto.unique_name, consumers)

@BeamTransformFactory.register_urn(python_urns.IMPULSE_READ_TRANSFORM, beam_runner_api_pb2.ReadPayload)
def create_read_from_impulse_python(factory, transform_id, transform_proto, parameter, consumers):
    if False:
        for i in range(10):
            print('nop')
    return operations.ImpulseReadOperation(common.NameContext(transform_proto.unique_name, transform_id), factory.counter_factory, factory.state_sampler, consumers, iobase.BoundedSource.from_runner_api(parameter.source, factory.context), factory.get_only_output_coder(transform_proto))

@BeamTransformFactory.register_urn(OLD_DATAFLOW_RUNNER_HARNESS_PARDO_URN, None)
def create_dofn_javasdk(factory, transform_id, transform_proto, serialized_fn, consumers):
    if False:
        print('Hello World!')
    return _create_pardo_operation(factory, transform_id, transform_proto, consumers, serialized_fn)

@BeamTransformFactory.register_urn(common_urns.sdf_components.PAIR_WITH_RESTRICTION.urn, beam_runner_api_pb2.ParDoPayload)
def create_pair_with_restriction(*args):
    if False:
        print('Hello World!')

    class PairWithRestriction(beam.DoFn):

        def __init__(self, fn, restriction_provider, watermark_estimator_provider):
            if False:
                print('Hello World!')
            self.restriction_provider = restriction_provider
            self.watermark_estimator_provider = watermark_estimator_provider

        def process(self, element, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            initial_restriction = self.restriction_provider.initial_restriction(element)
            initial_estimator_state = self.watermark_estimator_provider.initial_estimator_state(element, initial_restriction)
            yield (element, (initial_restriction, initial_estimator_state))
    return _create_sdf_operation(PairWithRestriction, *args)

@BeamTransformFactory.register_urn(common_urns.sdf_components.SPLIT_AND_SIZE_RESTRICTIONS.urn, beam_runner_api_pb2.ParDoPayload)
def create_split_and_size_restrictions(*args):
    if False:
        for i in range(10):
            print('nop')

    class SplitAndSizeRestrictions(beam.DoFn):

        def __init__(self, fn, restriction_provider, watermark_estimator_provider):
            if False:
                i = 10
                return i + 15
            self.restriction_provider = restriction_provider
            self.watermark_estimator_provider = watermark_estimator_provider

        def process(self, element_restriction, *args, **kwargs):
            if False:
                while True:
                    i = 10
            (element, (restriction, _)) = element_restriction
            for (part, size) in self.restriction_provider.split_and_size(element, restriction):
                if size < 0:
                    raise ValueError('Expected size >= 0 but received %s.' % size)
                estimator_state = self.watermark_estimator_provider.initial_estimator_state(element, part)
                yield ((element, (part, estimator_state)), size)
    return _create_sdf_operation(SplitAndSizeRestrictions, *args)

@BeamTransformFactory.register_urn(common_urns.sdf_components.TRUNCATE_SIZED_RESTRICTION.urn, beam_runner_api_pb2.ParDoPayload)
def create_truncate_sized_restriction(*args):
    if False:
        return 10

    class TruncateAndSizeRestriction(beam.DoFn):

        def __init__(self, fn, restriction_provider, watermark_estimator_provider):
            if False:
                while True:
                    i = 10
            self.restriction_provider = restriction_provider

        def process(self, element_restriction, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            ((element, (restriction, estimator_state)), _) = element_restriction
            truncated_restriction = self.restriction_provider.truncate(element, restriction)
            if truncated_restriction:
                truncated_restriction_size = self.restriction_provider.restriction_size(element, truncated_restriction)
                if truncated_restriction_size < 0:
                    raise ValueError('Expected size >= 0 but received %s.' % truncated_restriction_size)
                yield ((element, (truncated_restriction, estimator_state)), truncated_restriction_size)
    return _create_sdf_operation(TruncateAndSizeRestriction, *args, operation_cls=operations.SdfTruncateSizedRestrictions)

@BeamTransformFactory.register_urn(common_urns.sdf_components.PROCESS_SIZED_ELEMENTS_AND_RESTRICTIONS.urn, beam_runner_api_pb2.ParDoPayload)
def create_process_sized_elements_and_restrictions(factory, transform_id, transform_proto, parameter, consumers):
    if False:
        print('Hello World!')
    return _create_pardo_operation(factory, transform_id, transform_proto, consumers, core.DoFnInfo.from_runner_api(parameter.do_fn, factory.context).serialized_dofn_data(), parameter, operation_cls=operations.SdfProcessSizedElements)

def _create_sdf_operation(proxy_dofn, factory, transform_id, transform_proto, parameter, consumers, operation_cls=operations.DoOperation):
    if False:
        while True:
            i = 10
    dofn_data = pickler.loads(parameter.do_fn.payload)
    dofn = dofn_data[0]
    restriction_provider = common.DoFnSignature(dofn).get_restriction_provider()
    watermark_estimator_provider = common.DoFnSignature(dofn).get_watermark_estimator_provider()
    serialized_fn = pickler.dumps((proxy_dofn(dofn, restriction_provider, watermark_estimator_provider),) + dofn_data[1:])
    return _create_pardo_operation(factory, transform_id, transform_proto, consumers, serialized_fn, parameter, operation_cls=operation_cls)

@BeamTransformFactory.register_urn(common_urns.primitives.PAR_DO.urn, beam_runner_api_pb2.ParDoPayload)
def create_par_do(factory, transform_id, transform_proto, parameter, consumers):
    if False:
        for i in range(10):
            print('nop')
    return _create_pardo_operation(factory, transform_id, transform_proto, consumers, core.DoFnInfo.from_runner_api(parameter.do_fn, factory.context).serialized_dofn_data(), parameter)

def _create_pardo_operation(factory, transform_id, transform_proto, consumers, serialized_fn, pardo_proto=None, operation_cls=operations.DoOperation):
    if False:
        print('Hello World!')
    if pardo_proto and pardo_proto.side_inputs:
        input_tags_to_coders = factory.get_input_coders(transform_proto)
        tagged_side_inputs = [(tag, beam.pvalue.SideInputData.from_runner_api(si, factory.context)) for (tag, si) in pardo_proto.side_inputs.items()]
        tagged_side_inputs.sort(key=lambda tag_si: sideinputs.get_sideinput_index(tag_si[0]))
        side_input_maps = [StateBackedSideInputMap(factory.state_handler, transform_id, tag, si, input_tags_to_coders[tag]) for (tag, si) in tagged_side_inputs]
    else:
        side_input_maps = []
    output_tags = list(transform_proto.outputs.keys())
    dofn_data = pickler.loads(serialized_fn)
    if not dofn_data[-1]:
        if pardo_proto:
            other_input_tags = set.union(set(pardo_proto.side_inputs), set(pardo_proto.timer_family_specs))
        else:
            other_input_tags = ()
        (pcoll_id,) = [pcoll for (tag, pcoll) in transform_proto.inputs.items() if tag not in other_input_tags]
        windowing = factory.context.windowing_strategies.get_by_id(factory.descriptor.pcollections[pcoll_id].windowing_strategy_id)
        serialized_fn = pickler.dumps(dofn_data[:-1] + (windowing,))
    if pardo_proto and (pardo_proto.timer_family_specs or pardo_proto.state_specs or pardo_proto.restriction_coder_id):
        found_input_coder = None
        for (tag, pcoll_id) in transform_proto.inputs.items():
            if tag in pardo_proto.side_inputs:
                pass
            else:
                assert found_input_coder is None
                main_input_tag = tag
                found_input_coder = factory.get_windowed_coder(pcoll_id)
        assert found_input_coder is not None
        main_input_coder = found_input_coder
        if pardo_proto.timer_family_specs or pardo_proto.state_specs:
            user_state_context = FnApiUserStateContext(factory.state_handler, transform_id, main_input_coder.key_coder(), main_input_coder.window_coder)
        else:
            user_state_context = None
    else:
        user_state_context = None
    output_coders = factory.get_output_coders(transform_proto)
    spec = operation_specs.WorkerDoFn(serialized_fn=serialized_fn, output_tags=output_tags, input=None, side_inputs=None, output_coders=[output_coders[tag] for tag in output_tags])
    result = factory.augment_oldstyle_op(operation_cls(common.NameContext(transform_proto.unique_name, transform_id), spec, factory.counter_factory, factory.state_sampler, side_input_maps, user_state_context), transform_proto.unique_name, consumers, output_tags)
    if pardo_proto and pardo_proto.restriction_coder_id:
        result.input_info = operations.OpInputInfo(transform_id, main_input_tag, main_input_coder, transform_proto.outputs.keys())
    return result

def _create_simple_pardo_operation(factory, transform_id, transform_proto, consumers, dofn):
    if False:
        i = 10
        return i + 15
    serialized_fn = pickler.dumps((dofn, (), {}, [], None))
    return _create_pardo_operation(factory, transform_id, transform_proto, consumers, serialized_fn)

@BeamTransformFactory.register_urn(common_urns.primitives.ASSIGN_WINDOWS.urn, beam_runner_api_pb2.WindowingStrategy)
def create_assign_windows(factory, transform_id, transform_proto, parameter, consumers):
    if False:
        print('Hello World!')

    class WindowIntoDoFn(beam.DoFn):

        def __init__(self, windowing):
            if False:
                for i in range(10):
                    print('nop')
            self.windowing = windowing

        def process(self, element, timestamp=beam.DoFn.TimestampParam, window=beam.DoFn.WindowParam):
            if False:
                i = 10
                return i + 15
            new_windows = self.windowing.windowfn.assign(WindowFn.AssignContext(timestamp, element=element, window=window))
            yield WindowedValue(element, timestamp, new_windows)
    from apache_beam.transforms.core import Windowing
    from apache_beam.transforms.window import WindowFn
    windowing = Windowing.from_runner_api(parameter, factory.context)
    return _create_simple_pardo_operation(factory, transform_id, transform_proto, consumers, WindowIntoDoFn(windowing))

@BeamTransformFactory.register_urn(IDENTITY_DOFN_URN, None)
def create_identity_dofn(factory, transform_id, transform_proto, parameter, consumers):
    if False:
        print('Hello World!')
    return factory.augment_oldstyle_op(operations.FlattenOperation(common.NameContext(transform_proto.unique_name, transform_id), operation_specs.WorkerFlatten(None, [factory.get_only_output_coder(transform_proto)]), factory.counter_factory, factory.state_sampler), transform_proto.unique_name, consumers)

@BeamTransformFactory.register_urn(common_urns.combine_components.COMBINE_PER_KEY_PRECOMBINE.urn, beam_runner_api_pb2.CombinePayload)
def create_combine_per_key_precombine(factory, transform_id, transform_proto, payload, consumers):
    if False:
        i = 10
        return i + 15
    serialized_combine_fn = pickler.dumps((beam.CombineFn.from_runner_api(payload.combine_fn, factory.context), [], {}))
    return factory.augment_oldstyle_op(operations.PGBKCVOperation(common.NameContext(transform_proto.unique_name, transform_id), operation_specs.WorkerPartialGroupByKey(serialized_combine_fn, None, [factory.get_only_output_coder(transform_proto)]), factory.counter_factory, factory.state_sampler, factory.get_input_windowing(transform_proto)), transform_proto.unique_name, consumers)

@BeamTransformFactory.register_urn(common_urns.combine_components.COMBINE_PER_KEY_MERGE_ACCUMULATORS.urn, beam_runner_api_pb2.CombinePayload)
def create_combbine_per_key_merge_accumulators(factory, transform_id, transform_proto, payload, consumers):
    if False:
        print('Hello World!')
    return _create_combine_phase_operation(factory, transform_id, transform_proto, payload, consumers, 'merge')

@BeamTransformFactory.register_urn(common_urns.combine_components.COMBINE_PER_KEY_EXTRACT_OUTPUTS.urn, beam_runner_api_pb2.CombinePayload)
def create_combine_per_key_extract_outputs(factory, transform_id, transform_proto, payload, consumers):
    if False:
        while True:
            i = 10
    return _create_combine_phase_operation(factory, transform_id, transform_proto, payload, consumers, 'extract')

@BeamTransformFactory.register_urn(common_urns.combine_components.COMBINE_PER_KEY_CONVERT_TO_ACCUMULATORS.urn, beam_runner_api_pb2.CombinePayload)
def create_combine_per_key_convert_to_accumulators(factory, transform_id, transform_proto, payload, consumers):
    if False:
        for i in range(10):
            print('nop')
    return _create_combine_phase_operation(factory, transform_id, transform_proto, payload, consumers, 'convert')

@BeamTransformFactory.register_urn(common_urns.combine_components.COMBINE_GROUPED_VALUES.urn, beam_runner_api_pb2.CombinePayload)
def create_combine_grouped_values(factory, transform_id, transform_proto, payload, consumers):
    if False:
        while True:
            i = 10
    return _create_combine_phase_operation(factory, transform_id, transform_proto, payload, consumers, 'all')

def _create_combine_phase_operation(factory, transform_id, transform_proto, payload, consumers, phase):
    if False:
        for i in range(10):
            print('nop')
    serialized_combine_fn = pickler.dumps((beam.CombineFn.from_runner_api(payload.combine_fn, factory.context), [], {}))
    return factory.augment_oldstyle_op(operations.CombineOperation(common.NameContext(transform_proto.unique_name, transform_id), operation_specs.WorkerCombineFn(serialized_combine_fn, phase, None, [factory.get_only_output_coder(transform_proto)]), factory.counter_factory, factory.state_sampler), transform_proto.unique_name, consumers)

@BeamTransformFactory.register_urn(common_urns.primitives.FLATTEN.urn, None)
def create_flatten(factory, transform_id, transform_proto, payload, consumers):
    if False:
        return 10
    return factory.augment_oldstyle_op(operations.FlattenOperation(common.NameContext(transform_proto.unique_name, transform_id), operation_specs.WorkerFlatten(None, [factory.get_only_output_coder(transform_proto)]), factory.counter_factory, factory.state_sampler), transform_proto.unique_name, consumers)

@BeamTransformFactory.register_urn(common_urns.primitives.MAP_WINDOWS.urn, beam_runner_api_pb2.FunctionSpec)
def create_map_windows(factory, transform_id, transform_proto, mapping_fn_spec, consumers):
    if False:
        i = 10
        return i + 15
    assert mapping_fn_spec.urn == python_urns.PICKLED_WINDOW_MAPPING_FN
    window_mapping_fn = pickler.loads(mapping_fn_spec.payload)

    class MapWindows(beam.DoFn):

        def process(self, element):
            if False:
                for i in range(10):
                    print('nop')
            (key, window) = element
            return [(key, window_mapping_fn(window))]
    return _create_simple_pardo_operation(factory, transform_id, transform_proto, consumers, MapWindows())

@BeamTransformFactory.register_urn(common_urns.primitives.MERGE_WINDOWS.urn, beam_runner_api_pb2.FunctionSpec)
def create_merge_windows(factory, transform_id, transform_proto, mapping_fn_spec, consumers):
    if False:
        i = 10
        return i + 15
    assert mapping_fn_spec.urn == python_urns.PICKLED_WINDOWFN
    window_fn = pickler.loads(mapping_fn_spec.payload)

    class MergeWindows(beam.DoFn):

        def process(self, element):
            if False:
                print('Hello World!')
            (nonce, windows) = element
            original_windows = set(windows)
            merged_windows = collections.defaultdict(set)

            class RecordingMergeContext(window.WindowFn.MergeContext):

                def merge(self, to_be_merged, merge_result):
                    if False:
                        i = 10
                        return i + 15
                    originals = merged_windows[merge_result]
                    for window in to_be_merged:
                        if window in original_windows:
                            originals.add(window)
                            original_windows.remove(window)
                        else:
                            originals.update(merged_windows.pop(window))
            window_fn.merge(RecordingMergeContext(windows))
            yield (nonce, (original_windows, merged_windows.items()))
    return _create_simple_pardo_operation(factory, transform_id, transform_proto, consumers, MergeWindows())

@BeamTransformFactory.register_urn(common_urns.primitives.TO_STRING.urn, None)
def create_to_string_fn(factory, transform_id, transform_proto, mapping_fn_spec, consumers):
    if False:
        for i in range(10):
            print('nop')

    class ToString(beam.DoFn):

        def process(self, element):
            if False:
                i = 10
                return i + 15
            (key, value) = element
            return [(key, str(value))]
    return _create_simple_pardo_operation(factory, transform_id, transform_proto, consumers, ToString())