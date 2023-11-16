"""SDK harness for executing Python Fns via the Fn API."""
import abc
import collections
import contextlib
import functools
import json
import logging
import queue
import sys
import threading
import time
import traceback
from concurrent import futures
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import DefaultDict
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
import grpc
from apache_beam.coders import coder_impl
from apache_beam.metrics import monitoring_infos
from apache_beam.metrics.execution import MetricsEnvironment
from apache_beam.portability.api import beam_fn_api_pb2
from apache_beam.portability.api import beam_fn_api_pb2_grpc
from apache_beam.portability.api import metrics_pb2
from apache_beam.runners.worker import bundle_processor
from apache_beam.runners.worker import data_plane
from apache_beam.runners.worker import data_sampler
from apache_beam.runners.worker import statesampler
from apache_beam.runners.worker.channel_factory import GRPCChannelFactory
from apache_beam.runners.worker.data_plane import PeriodicThread
from apache_beam.runners.worker.statecache import CacheAware
from apache_beam.runners.worker.statecache import StateCache
from apache_beam.runners.worker.worker_id_interceptor import WorkerIdInterceptor
from apache_beam.runners.worker.worker_status import FnApiWorkerStatusHandler
from apache_beam.utils import thread_pool_executor
from apache_beam.utils.sentinel import Sentinel
if TYPE_CHECKING:
    from apache_beam.portability.api import endpoints_pb2
    from apache_beam.utils.profiler import Profile
T = TypeVar('T')
_KT = TypeVar('_KT')
_VT = TypeVar('_VT')
_LOGGER = logging.getLogger(__name__)
DEFAULT_BUNDLE_PROCESSOR_CACHE_SHUTDOWN_THRESHOLD_S = 60
MAX_KNOWN_NOT_RUNNING_INSTRUCTIONS = 1000
MAX_FAILED_INSTRUCTIONS = 10000
_GRPC_SERVICE_CONFIG = json.dumps({'methodConfig': [{'name': [{'service': 'org.apache.beam.model.fn_execution.v1.BeamFnState'}], 'retryPolicy': {'maxAttempts': 5, 'initialBackoff': '0.1s', 'maxBackoff': '5s', 'backoffMultiplier': 2, 'retryableStatusCodes': ['UNAVAILABLE']}}]})

class ShortIdCache(object):
    """ Cache for MonitoringInfo "short ids"
  """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._lock = threading.Lock()
        self._last_short_id = 0
        self._info_key_to_short_id = {}
        self._short_id_to_info = {}

    def get_short_id(self, monitoring_info):
        if False:
            while True:
                i = 10
        ' Returns the assigned shortId for a given MonitoringInfo, assigns one if\n    not assigned already.\n    '
        key = monitoring_infos.to_key(monitoring_info)
        with self._lock:
            try:
                return self._info_key_to_short_id[key]
            except KeyError:
                self._last_short_id += 1
                shortId = hex(self._last_short_id)[2:]
                payload_cleared = metrics_pb2.MonitoringInfo()
                payload_cleared.CopyFrom(monitoring_info)
                payload_cleared.ClearField('payload')
                self._info_key_to_short_id[key] = shortId
                self._short_id_to_info[shortId] = payload_cleared
                return shortId

    def get_infos(self, short_ids):
        if False:
            for i in range(10):
                print('nop')
        ' Gets the base MonitoringInfo (with payload cleared) for each short ID.\n\n    Throws KeyError if an unassigned short ID is encountered.\n    '
        return {short_id: self._short_id_to_info[short_id] for short_id in short_ids}
SHORT_ID_CACHE = ShortIdCache()

class SdkHarness(object):
    REQUEST_METHOD_PREFIX = '_request_'

    def __init__(self, control_address, credentials=None, worker_id=None, state_cache_size=0, data_buffer_time_limit_ms=0, profiler_factory=None, status_address=None, enable_heap_dump=False, data_sampler=None, deferred_exception=None):
        if False:
            i = 10
            return i + 15
        self._alive = True
        self._worker_index = 0
        self._worker_id = worker_id
        self._state_cache = StateCache(state_cache_size)
        self._deferred_exception = deferred_exception
        options = [('grpc.max_receive_message_length', -1), ('grpc.max_send_message_length', -1)]
        if credentials is None:
            _LOGGER.info('Creating insecure control channel for %s.', control_address)
            self._control_channel = GRPCChannelFactory.insecure_channel(control_address, options=options)
        else:
            _LOGGER.info('Creating secure control channel for %s.', control_address)
            self._control_channel = GRPCChannelFactory.secure_channel(control_address, credentials, options=options)
        grpc.channel_ready_future(self._control_channel).result(timeout=60)
        _LOGGER.info('Control channel established.')
        self._control_channel = grpc.intercept_channel(self._control_channel, WorkerIdInterceptor(self._worker_id))
        self._data_channel_factory = data_plane.GrpcClientDataChannelFactory(credentials, self._worker_id, data_buffer_time_limit_ms)
        self._state_handler_factory = GrpcStateHandlerFactory(self._state_cache, credentials)
        self._profiler_factory = profiler_factory
        self.data_sampler = data_sampler

        def default_factory(id):
            if False:
                i = 10
                return i + 15
            return self._control_stub.GetProcessBundleDescriptor(beam_fn_api_pb2.GetProcessBundleDescriptorRequest(process_bundle_descriptor_id=id))
        self._fns = KeyedDefaultDict(default_factory)
        self._bundle_processor_cache = BundleProcessorCache(state_handler_factory=self._state_handler_factory, data_channel_factory=self._data_channel_factory, fns=self._fns, data_sampler=self.data_sampler)
        if status_address:
            try:
                self._status_handler = FnApiWorkerStatusHandler(status_address, self._bundle_processor_cache, self._state_cache, enable_heap_dump)
            except Exception:
                traceback_string = traceback.format_exc()
                _LOGGER.warning('Error creating worker status request handler, skipping status report. Trace back: %s' % traceback_string)
        else:
            self._status_handler = None
        self._report_progress_executor = futures.ThreadPoolExecutor(max_workers=1)
        self._worker_thread_pool = thread_pool_executor.shared_unbounded_instance()
        self._responses = queue.Queue()
        _LOGGER.info('Initializing SDKHarness with unbounded number of workers.')

    def run(self):
        if False:
            return 10
        self._control_stub = beam_fn_api_pb2_grpc.BeamFnControlStub(self._control_channel)
        no_more_work = Sentinel.sentinel

        def get_responses():
            if False:
                while True:
                    i = 10
            while True:
                response = self._responses.get()
                if response is no_more_work:
                    return
                yield response
        self._alive = True
        try:
            for work_request in self._control_stub.Control(get_responses()):
                _LOGGER.debug('Got work %s', work_request.instruction_id)
                request_type = work_request.WhichOneof('request')
                getattr(self, SdkHarness.REQUEST_METHOD_PREFIX + request_type)(work_request)
        finally:
            self._alive = False
            if self.data_sampler:
                self.data_sampler.stop()
        _LOGGER.info('No more requests from control plane')
        _LOGGER.info('SDK Harness waiting for in-flight requests to complete')
        self._worker_thread_pool.shutdown()
        self._responses.put(no_more_work)
        self._data_channel_factory.close()
        self._state_handler_factory.close()
        self._bundle_processor_cache.shutdown()
        if self._status_handler:
            self._status_handler.close()
        _LOGGER.info('Done consuming work.')

    def _execute(self, task, request):
        if False:
            i = 10
            return i + 15
        with statesampler.instruction_id(request.instruction_id):
            try:
                response = task()
            except:
                traceback_string = traceback.format_exc()
                print(traceback_string, file=sys.stderr)
                _LOGGER.error('Error processing instruction %s. Original traceback is\n%s\n', request.instruction_id, traceback_string)
                response = beam_fn_api_pb2.InstructionResponse(instruction_id=request.instruction_id, error=traceback_string)
            self._responses.put(response)

    def _request_register(self, request):
        if False:
            i = 10
            return i + 15
        self._execute(lambda : self.create_worker().do_instruction(request), request)

    def _request_process_bundle(self, request):
        if False:
            print('Hello World!')
        if self._deferred_exception:
            raise self._deferred_exception
        self._bundle_processor_cache.activate(request.instruction_id)
        self._request_execute(request)

    def _request_process_bundle_split(self, request):
        if False:
            return 10
        self._request_process_bundle_action(request)

    def _request_process_bundle_progress(self, request):
        if False:
            print('Hello World!')
        self._request_process_bundle_action(request)

    def _request_process_bundle_action(self, request):
        if False:
            for i in range(10):
                print('nop')

        def task():
            if False:
                return 10
            self._execute(lambda : self.create_worker().do_instruction(request), request)
        self._report_progress_executor.submit(task)

    def _request_finalize_bundle(self, request):
        if False:
            return 10
        self._request_execute(request)

    def _request_harness_monitoring_infos(self, request):
        if False:
            for i in range(10):
                print('nop')
        process_wide_monitoring_infos = MetricsEnvironment.process_wide_container().to_runner_api_monitoring_infos(None).values()
        self._execute(lambda : beam_fn_api_pb2.InstructionResponse(instruction_id=request.instruction_id, harness_monitoring_infos=beam_fn_api_pb2.HarnessMonitoringInfosResponse(monitoring_data={SHORT_ID_CACHE.get_short_id(info): info.payload for info in process_wide_monitoring_infos})), request)

    def _request_monitoring_infos(self, request):
        if False:
            while True:
                i = 10
        self._execute(lambda : beam_fn_api_pb2.InstructionResponse(instruction_id=request.instruction_id, monitoring_infos=beam_fn_api_pb2.MonitoringInfosMetadataResponse(monitoring_info=SHORT_ID_CACHE.get_infos(request.monitoring_infos.monitoring_info_id))), request)

    def _request_execute(self, request):
        if False:
            return 10

        def task():
            if False:
                while True:
                    i = 10
            self._execute(lambda : self.create_worker().do_instruction(request), request)
        self._worker_thread_pool.submit(task)
        _LOGGER.debug('Currently using %s threads.' % len(self._worker_thread_pool._workers))

    def _request_sample_data(self, request):
        if False:
            for i in range(10):
                print('nop')

        def get_samples(request):
            if False:
                return 10
            samples = beam_fn_api_pb2.SampleDataResponse()
            if self.data_sampler is not None:
                samples = self.data_sampler.samples(request.sample_data.pcollection_ids)
            return beam_fn_api_pb2.InstructionResponse(instruction_id=request.instruction_id, sample_data=samples)
        self._execute(lambda : get_samples(request), request)

    def create_worker(self):
        if False:
            return 10
        return SdkWorker(self._bundle_processor_cache, profiler_factory=self._profiler_factory)

class BundleProcessorCache(object):
    """A cache for ``BundleProcessor``s.

  ``BundleProcessor`` objects are cached by the id of their
  ``beam_fn_api_pb2.ProcessBundleDescriptor``.

  Attributes:
    fns (dict): A dictionary that maps bundle descriptor IDs to instances of
      ``beam_fn_api_pb2.ProcessBundleDescriptor``.
    state_handler_factory (``StateHandlerFactory``): Used to create state
      handlers to be used by a ``bundle_processor.BundleProcessor`` during
      processing.
    data_channel_factory (``data_plane.DataChannelFactory``)
    active_bundle_processors (dict): A dictionary, indexed by instruction IDs,
      containing ``bundle_processor.BundleProcessor`` objects that are currently
      active processing the corresponding instruction.
    cached_bundle_processors (dict): A dictionary, indexed by bundle processor
      id, of cached ``bundle_processor.BundleProcessor`` that are not currently
      performing processing.
  """
    periodic_shutdown = None

    def __init__(self, state_handler_factory, data_channel_factory, fns, data_sampler=None):
        if False:
            for i in range(10):
                print('nop')
        self.fns = fns
        self.state_handler_factory = state_handler_factory
        self.data_channel_factory = data_channel_factory
        self.known_not_running_instruction_ids = collections.OrderedDict()
        self.failed_instruction_ids = collections.OrderedDict()
        self.active_bundle_processors = {}
        self.cached_bundle_processors = collections.defaultdict(list)
        self.last_access_times = collections.defaultdict(float)
        self._schedule_periodic_shutdown()
        self._lock = threading.Lock()
        self.data_sampler = data_sampler

    def register(self, bundle_descriptor):
        if False:
            for i in range(10):
                print('nop')
        'Register a ``beam_fn_api_pb2.ProcessBundleDescriptor`` by its id.'
        self.fns[bundle_descriptor.id] = bundle_descriptor

    def activate(self, instruction_id):
        if False:
            print('Hello World!')
        'Makes the ``instruction_id`` known to the bundle processor.\n\n    Allows ``lookup`` to return ``None``. Necessary if ``lookup`` can occur\n    before ``get``.\n    '
        with self._lock:
            self.known_not_running_instruction_ids[instruction_id] = True

    def get(self, instruction_id, bundle_descriptor_id):
        if False:
            print('Hello World!')
        '\n    Return the requested ``BundleProcessor``, creating it if necessary.\n\n    Moves the ``BundleProcessor`` from the inactive to the active cache.\n    '
        with self._lock:
            try:
                processor = self.cached_bundle_processors[bundle_descriptor_id].pop()
                self.active_bundle_processors[instruction_id] = (bundle_descriptor_id, processor)
                try:
                    del self.known_not_running_instruction_ids[instruction_id]
                except KeyError:
                    pass
                return processor
            except IndexError:
                pass
        processor = bundle_processor.BundleProcessor(self.fns[bundle_descriptor_id], self.state_handler_factory.create_state_handler(self.fns[bundle_descriptor_id].state_api_service_descriptor), self.data_channel_factory, self.data_sampler)
        with self._lock:
            self.active_bundle_processors[instruction_id] = (bundle_descriptor_id, processor)
            try:
                del self.known_not_running_instruction_ids[instruction_id]
            except KeyError:
                pass
        return processor

    def lookup(self, instruction_id):
        if False:
            i = 10
            return i + 15
        '\n    Return the requested ``BundleProcessor`` from the cache.\n\n    Will return ``None`` if the BundleProcessor is known but not yet ready. Will\n    raise an error if the ``instruction_id`` is not known or has been discarded.\n    '
        with self._lock:
            if instruction_id in self.failed_instruction_ids:
                raise RuntimeError('Bundle processing associated with %s has failed. Check prior failing response for details.' % instruction_id)
            processor = self.active_bundle_processors.get(instruction_id, (None, None))[-1]
            if processor:
                return processor
            if instruction_id in self.known_not_running_instruction_ids:
                return None
            raise RuntimeError('Unknown process bundle id %s.' % instruction_id)

    def discard(self, instruction_id):
        if False:
            return 10
        '\n    Marks the instruction id as failed shutting down the ``BundleProcessor``.\n    '
        with self._lock:
            self.failed_instruction_ids[instruction_id] = True
            while len(self.failed_instruction_ids) > MAX_FAILED_INSTRUCTIONS:
                self.failed_instruction_ids.popitem(last=False)
            processor = self.active_bundle_processors[instruction_id][1]
            del self.active_bundle_processors[instruction_id]
        processor.shutdown()

    def release(self, instruction_id):
        if False:
            i = 10
            return i + 15
        '\n    Release the requested ``BundleProcessor``.\n\n    Resets the ``BundleProcessor`` and moves it from the active to the\n    inactive cache.\n    '
        with self._lock:
            self.known_not_running_instruction_ids[instruction_id] = True
            while len(self.known_not_running_instruction_ids) > MAX_KNOWN_NOT_RUNNING_INSTRUCTIONS:
                self.known_not_running_instruction_ids.popitem(last=False)
            (descriptor_id, processor) = self.active_bundle_processors.pop(instruction_id)
        processor.reset()
        with self._lock:
            self.last_access_times[descriptor_id] = time.time()
            self.cached_bundle_processors[descriptor_id].append(processor)

    def shutdown(self):
        if False:
            while True:
                i = 10
        '\n    Shutdown all ``BundleProcessor``s in the cache.\n    '
        if self.periodic_shutdown:
            self.periodic_shutdown.cancel()
            self.periodic_shutdown.join()
            self.periodic_shutdown = None
        for instruction_id in list(self.active_bundle_processors.keys()):
            self.discard(instruction_id)
        for cached_bundle_processors in self.cached_bundle_processors.values():
            BundleProcessorCache._shutdown_cached_bundle_processors(cached_bundle_processors)

    def _schedule_periodic_shutdown(self):
        if False:
            i = 10
            return i + 15

        def shutdown_inactive_bundle_processors():
            if False:
                i = 10
                return i + 15
            for (descriptor_id, last_access_time) in self.last_access_times.items():
                if time.time() - last_access_time > DEFAULT_BUNDLE_PROCESSOR_CACHE_SHUTDOWN_THRESHOLD_S:
                    BundleProcessorCache._shutdown_cached_bundle_processors(self.cached_bundle_processors[descriptor_id])
        self.periodic_shutdown = PeriodicThread(DEFAULT_BUNDLE_PROCESSOR_CACHE_SHUTDOWN_THRESHOLD_S, shutdown_inactive_bundle_processors)
        self.periodic_shutdown.daemon = True
        self.periodic_shutdown.start()

    @staticmethod
    def _shutdown_cached_bundle_processors(cached_bundle_processors):
        if False:
            i = 10
            return i + 15
        try:
            while True:
                bundle_processor = cached_bundle_processors.pop()
                bundle_processor.shutdown()
        except IndexError:
            pass

class SdkWorker(object):

    def __init__(self, bundle_processor_cache, profiler_factory=None):
        if False:
            i = 10
            return i + 15
        self.bundle_processor_cache = bundle_processor_cache
        self.profiler_factory = profiler_factory

    def do_instruction(self, request):
        if False:
            print('Hello World!')
        request_type = request.WhichOneof('request')
        if request_type:
            return getattr(self, request_type)(getattr(request, request_type), request.instruction_id)
        else:
            raise NotImplementedError

    def register(self, request, instruction_id):
        if False:
            print('Hello World!')
        'Registers a set of ``beam_fn_api_pb2.ProcessBundleDescriptor``s.\n\n    This set of ``beam_fn_api_pb2.ProcessBundleDescriptor`` come as part of a\n    ``beam_fn_api_pb2.RegisterRequest``, which the runner sends to the SDK\n    worker before starting processing to register stages.\n    '
        for process_bundle_descriptor in request.process_bundle_descriptor:
            self.bundle_processor_cache.register(process_bundle_descriptor)
        return beam_fn_api_pb2.InstructionResponse(instruction_id=instruction_id, register=beam_fn_api_pb2.RegisterResponse())

    def process_bundle(self, request, instruction_id):
        if False:
            for i in range(10):
                print('nop')
        bundle_processor = self.bundle_processor_cache.get(instruction_id, request.process_bundle_descriptor_id)
        try:
            with bundle_processor.state_handler.process_instruction_id(instruction_id, request.cache_tokens):
                with self.maybe_profile(instruction_id):
                    (delayed_applications, requests_finalization) = bundle_processor.process_bundle(instruction_id)
                    monitoring_infos = bundle_processor.monitoring_infos()
                    response = beam_fn_api_pb2.InstructionResponse(instruction_id=instruction_id, process_bundle=beam_fn_api_pb2.ProcessBundleResponse(residual_roots=delayed_applications, monitoring_infos=monitoring_infos, monitoring_data={SHORT_ID_CACHE.get_short_id(info): info.payload for info in monitoring_infos}, requires_finalization=requests_finalization))
            if not requests_finalization:
                self.bundle_processor_cache.release(instruction_id)
            return response
        except:
            self.bundle_processor_cache.discard(instruction_id)
            raise

    def process_bundle_split(self, request, instruction_id):
        if False:
            return 10
        try:
            processor = self.bundle_processor_cache.lookup(request.instruction_id)
        except RuntimeError:
            return beam_fn_api_pb2.InstructionResponse(instruction_id=instruction_id, error=traceback.format_exc())
        process_bundle_split = processor.try_split(request) if processor else beam_fn_api_pb2.ProcessBundleSplitResponse()
        return beam_fn_api_pb2.InstructionResponse(instruction_id=instruction_id, process_bundle_split=process_bundle_split)

    def process_bundle_progress(self, request, instruction_id):
        if False:
            print('Hello World!')
        try:
            processor = self.bundle_processor_cache.lookup(request.instruction_id)
        except RuntimeError:
            return beam_fn_api_pb2.InstructionResponse(instruction_id=instruction_id, error=traceback.format_exc())
        if processor:
            monitoring_infos = processor.monitoring_infos()
        else:
            monitoring_infos = []
        return beam_fn_api_pb2.InstructionResponse(instruction_id=instruction_id, process_bundle_progress=beam_fn_api_pb2.ProcessBundleProgressResponse(monitoring_infos=monitoring_infos, monitoring_data={SHORT_ID_CACHE.get_short_id(info): info.payload for info in monitoring_infos}))

    def finalize_bundle(self, request, instruction_id):
        if False:
            while True:
                i = 10
        try:
            processor = self.bundle_processor_cache.lookup(request.instruction_id)
        except RuntimeError:
            return beam_fn_api_pb2.InstructionResponse(instruction_id=instruction_id, error=traceback.format_exc())
        if processor:
            try:
                finalize_response = processor.finalize_bundle()
                self.bundle_processor_cache.release(request.instruction_id)
                return beam_fn_api_pb2.InstructionResponse(instruction_id=instruction_id, finalize_bundle=finalize_response)
            except:
                self.bundle_processor_cache.discard(request.instruction_id)
                raise
        raise RuntimeError('Bundle is not in a finalizable state for %s' % instruction_id)

    @contextlib.contextmanager
    def maybe_profile(self, instruction_id):
        if False:
            i = 10
            return i + 15
        if self.profiler_factory:
            profiler = self.profiler_factory(instruction_id)
            if profiler:
                with profiler:
                    yield
            else:
                yield
        else:
            yield

class StateHandler(metaclass=abc.ABCMeta):
    """An abstract object representing a ``StateHandler``."""

    @abc.abstractmethod
    def get_raw(self, state_key, continuation_token=None):
        if False:
            return 10
        'Gets the contents of state for the given state key.\n\n    State is associated to a state key, AND an instruction_id, which is set\n    when calling process_instruction_id.\n\n    Returns a tuple with the contents in state, and an optional continuation\n    token, which is used to page the API.\n    '
        raise NotImplementedError(type(self))

    @abc.abstractmethod
    def append_raw(self, state_key, data):
        if False:
            print('Hello World!')
        'Append the input data into the state key.\n\n    Returns a future that allows one to wait for the completion of the call.\n\n    State is associated to a state key, AND an instruction_id, which is set\n    when calling process_instruction_id.\n    '
        raise NotImplementedError(type(self))

    @abc.abstractmethod
    def clear(self, state_key):
        if False:
            print('Hello World!')
        'Clears the contents of a cell for the input state key.\n\n    Returns a future that allows one to wait for the completion of the call.\n\n    State is associated to a state key, AND an instruction_id, which is set\n    when calling process_instruction_id.\n    '
        raise NotImplementedError(type(self))

    @abc.abstractmethod
    @contextlib.contextmanager
    def process_instruction_id(self, bundle_id):
        if False:
            i = 10
            return i + 15
        'Switch the context of the state handler to a specific instruction.\n\n    This must be called before performing any write or read operations on the\n    existing state.\n    '
        raise NotImplementedError(type(self))

    @abc.abstractmethod
    def done(self):
        if False:
            while True:
                i = 10
        'Mark the state handler as done, and potentially delete all context.'
        raise NotImplementedError(type(self))

class StateHandlerFactory(metaclass=abc.ABCMeta):
    """An abstract factory for creating ``DataChannel``."""

    @abc.abstractmethod
    def create_state_handler(self, api_service_descriptor):
        if False:
            for i in range(10):
                print('nop')
        'Returns a ``StateHandler`` from the given ApiServiceDescriptor.'
        raise NotImplementedError(type(self))

    @abc.abstractmethod
    def close(self):
        if False:
            return 10
        'Close all channels that this factory owns.'
        raise NotImplementedError(type(self))

class GrpcStateHandlerFactory(StateHandlerFactory):
    """A factory for ``GrpcStateHandler``.

  Caches the created channels by ``state descriptor url``.
  """

    def __init__(self, state_cache, credentials=None):
        if False:
            i = 10
            return i + 15
        self._state_handler_cache = {}
        self._lock = threading.Lock()
        self._throwing_state_handler = ThrowingStateHandler()
        self._credentials = credentials
        self._state_cache = state_cache

    def create_state_handler(self, api_service_descriptor):
        if False:
            return 10
        if not api_service_descriptor:
            return self._throwing_state_handler
        url = api_service_descriptor.url
        if url not in self._state_handler_cache:
            with self._lock:
                if url not in self._state_handler_cache:
                    options = [('grpc.max_receive_message_length', -1), ('grpc.max_send_message_length', -1), ('grpc.service_config', _GRPC_SERVICE_CONFIG)]
                    if self._credentials is None:
                        _LOGGER.info('Creating insecure state channel for %s.', url)
                        grpc_channel = GRPCChannelFactory.insecure_channel(url, options=options)
                    else:
                        _LOGGER.info('Creating secure state channel for %s.', url)
                        grpc_channel = GRPCChannelFactory.secure_channel(url, self._credentials, options=options)
                    _LOGGER.info('State channel established.')
                    grpc_channel = grpc.intercept_channel(grpc_channel, WorkerIdInterceptor())
                    self._state_handler_cache[url] = GlobalCachingStateHandler(self._state_cache, GrpcStateHandler(beam_fn_api_pb2_grpc.BeamFnStateStub(grpc_channel)))
        return self._state_handler_cache[url]

    def close(self):
        if False:
            i = 10
            return i + 15
        _LOGGER.info('Closing all cached gRPC state handlers.')
        for (_, state_handler) in self._state_handler_cache.items():
            state_handler.done()
        self._state_handler_cache.clear()
        self._state_cache.invalidate_all()

class CachingStateHandler(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    @contextlib.contextmanager
    def process_instruction_id(self, bundle_id, cache_tokens):
        if False:
            print('Hello World!')
        raise NotImplementedError(type(self))

    @abc.abstractmethod
    def blocking_get(self, state_key, coder):
        if False:
            print('Hello World!')
        raise NotImplementedError(type(self))

    @abc.abstractmethod
    def extend(self, state_key, coder, elements):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError(type(self))

    @abc.abstractmethod
    def clear(self, state_key):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError(type(self))

    @abc.abstractmethod
    def done(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError(type(self))

class ThrowingStateHandler(CachingStateHandler):
    """A caching state handler that errors on any requests."""

    @contextlib.contextmanager
    def process_instruction_id(self, bundle_id, cache_tokens):
        if False:
            print('Hello World!')
        raise RuntimeError('Unable to handle state requests for ProcessBundleDescriptor for bundle id %s.' % bundle_id)

    def blocking_get(self, state_key, coder):
        if False:
            i = 10
            return i + 15
        raise RuntimeError('Unable to handle state requests for ProcessBundleDescriptor without state ApiServiceDescriptor for state key %s.' % state_key)

    def extend(self, state_key, coder, elements):
        if False:
            while True:
                i = 10
        raise RuntimeError('Unable to handle state requests for ProcessBundleDescriptor without state ApiServiceDescriptor for state key %s.' % state_key)

    def clear(self, state_key):
        if False:
            return 10
        raise RuntimeError('Unable to handle state requests for ProcessBundleDescriptor without state ApiServiceDescriptor for state key %s.' % state_key)

    def done(self):
        if False:
            print('Hello World!')
        raise RuntimeError('Unable to handle state requests for ProcessBundleDescriptor.')

class GrpcStateHandler(StateHandler):
    _DONE = Sentinel.sentinel

    def __init__(self, state_stub):
        if False:
            for i in range(10):
                print('nop')
        self._lock = threading.Lock()
        self._state_stub = state_stub
        self._requests = queue.Queue()
        self._responses_by_id = {}
        self._last_id = 0
        self._exception = None
        self._context = threading.local()
        self.start()

    @contextlib.contextmanager
    def process_instruction_id(self, bundle_id):
        if False:
            return 10
        if getattr(self._context, 'process_instruction_id', None) is not None:
            raise RuntimeError('Already bound to %r' % self._context.process_instruction_id)
        self._context.process_instruction_id = bundle_id
        try:
            yield
        finally:
            self._context.process_instruction_id = None

    def start(self):
        if False:
            while True:
                i = 10
        self._done = False

        def request_iter():
            if False:
                while True:
                    i = 10
            while True:
                request = self._requests.get()
                if request is self._DONE or self._done:
                    break
                yield request
        responses = self._state_stub.State(request_iter())

        def pull_responses():
            if False:
                while True:
                    i = 10
            try:
                for response in responses:
                    future = self._responses_by_id.pop(response.id)
                    future.set(response)
                    if self._done:
                        break
            except Exception as e:
                self._exception = e
                raise
        reader = threading.Thread(target=pull_responses, name='read_state')
        reader.daemon = True
        reader.start()

    def done(self):
        if False:
            return 10
        self._done = True
        self._requests.put(self._DONE)

    def get_raw(self, state_key, continuation_token=None):
        if False:
            return 10
        response = self._blocking_request(beam_fn_api_pb2.StateRequest(state_key=state_key, get=beam_fn_api_pb2.StateGetRequest(continuation_token=continuation_token)))
        return (response.get.data, response.get.continuation_token)

    def append_raw(self, state_key, data):
        if False:
            return 10
        return self._request(beam_fn_api_pb2.StateRequest(state_key=state_key, append=beam_fn_api_pb2.StateAppendRequest(data=data)))

    def clear(self, state_key):
        if False:
            for i in range(10):
                print('nop')
        return self._request(beam_fn_api_pb2.StateRequest(state_key=state_key, clear=beam_fn_api_pb2.StateClearRequest()))

    def _request(self, request):
        if False:
            for i in range(10):
                print('nop')
        request.id = self._next_id()
        request.instruction_id = self._context.process_instruction_id
        self._responses_by_id[request.id] = future = _Future[beam_fn_api_pb2.StateResponse]()
        self._requests.put(request)
        return future

    def _blocking_request(self, request):
        if False:
            print('Hello World!')
        req_future = self._request(request)
        while not req_future.wait(timeout=1):
            if self._exception:
                raise self._exception
            elif self._done:
                raise RuntimeError()
        response = req_future.get()
        if response.error:
            raise RuntimeError(response.error)
        else:
            return response

    def _next_id(self):
        if False:
            while True:
                i = 10
        with self._lock:
            self._last_id += 1
            request_id = self._last_id
        return str(request_id)

class GlobalCachingStateHandler(CachingStateHandler):
    """ A State handler which retrieves and caches state.
   If caching is activated, caches across bundles using a supplied cache token.
   If activated but no cache token is supplied, caching is done at the bundle
   level.
  """

    def __init__(self, global_state_cache, underlying_state):
        if False:
            while True:
                i = 10
        self._underlying = underlying_state
        self._state_cache = global_state_cache
        self._context = threading.local()

    @contextlib.contextmanager
    def process_instruction_id(self, bundle_id, cache_tokens):
        if False:
            return 10
        if getattr(self._context, 'user_state_cache_token', None) is not None:
            raise RuntimeError('Cache tokens already set to %s' % self._context.user_state_cache_token)
        self._context.side_input_cache_tokens = {}
        user_state_cache_token = None
        for cache_token_struct in cache_tokens:
            if cache_token_struct.HasField('user_state'):
                assert not user_state_cache_token
                user_state_cache_token = cache_token_struct.token
            elif cache_token_struct.HasField('side_input'):
                self._context.side_input_cache_tokens[cache_token_struct.side_input.transform_id, cache_token_struct.side_input.side_input_id] = cache_token_struct.token
        self._context.bundle_cache_token = bundle_id
        try:
            self._context.user_state_cache_token = user_state_cache_token
            with self._underlying.process_instruction_id(bundle_id):
                yield
        finally:
            self._context.side_input_cache_tokens = {}
            self._context.user_state_cache_token = None
            self._context.bundle_cache_token = None

    def blocking_get(self, state_key, coder):
        if False:
            i = 10
            return i + 15
        cache_token = self._get_cache_token(state_key)
        if not cache_token:
            return self._lazy_iterator(state_key, coder)
        cache_state_key = self._convert_to_cache_key(state_key)
        return self._state_cache.get((cache_state_key, cache_token), lambda key: self._partially_cached_iterable(state_key, coder))

    def extend(self, state_key, coder, elements):
        if False:
            return 10
        cache_token = self._get_cache_token(state_key)
        if cache_token:
            cache_key = self._convert_to_cache_key(state_key)
            cached_value = self._state_cache.peek((cache_key, cache_token))
            if isinstance(cached_value, list):
                elements = list(elements)
                cached_value.extend(elements)
                self._state_cache.put((cache_key, cache_token), cached_value)
        futures = []
        out = coder_impl.create_OutputStream()
        for element in elements:
            coder.encode_to_stream(element, out, True)
            if out.size() > data_plane._DEFAULT_SIZE_FLUSH_THRESHOLD:
                futures.append(self._underlying.append_raw(state_key, out.get()))
                out = coder_impl.create_OutputStream()
        if out.size():
            futures.append(self._underlying.append_raw(state_key, out.get()))
        return _DeferredCall(lambda *results: beam_fn_api_pb2.StateResponse(error='\n'.join((result.error for result in results if result and result.error)), append=beam_fn_api_pb2.StateAppendResponse()), *futures)

    def clear(self, state_key):
        if False:
            print('Hello World!')
        cache_token = self._get_cache_token(state_key)
        if cache_token:
            cache_key = self._convert_to_cache_key(state_key)
            self._state_cache.put((cache_key, cache_token), [])
        return self._underlying.clear(state_key)

    def done(self):
        if False:
            return 10
        self._underlying.done()

    def _lazy_iterator(self, state_key, coder, continuation_token=None):
        if False:
            return 10
        'Materializes the state lazily, one element at a time.\n       :return A generator which returns the next element if advanced.\n    '
        while True:
            (data, continuation_token) = self._underlying.get_raw(state_key, continuation_token)
            input_stream = coder_impl.create_InputStream(data)
            while input_stream.size() > 0:
                yield coder.decode_from_stream(input_stream, True)
            if not continuation_token:
                break

    def _get_cache_token(self, state_key):
        if False:
            print('Hello World!')
        if not self._state_cache.is_cache_enabled():
            return None
        elif state_key.HasField('bag_user_state'):
            if self._context.user_state_cache_token:
                return self._context.user_state_cache_token
            else:
                return self._context.bundle_cache_token
        elif state_key.WhichOneof('type').endswith('_side_input'):
            side_input = getattr(state_key, state_key.WhichOneof('type'))
            return self._context.side_input_cache_tokens.get((side_input.transform_id, side_input.side_input_id), self._context.bundle_cache_token)
        return None

    def _partially_cached_iterable(self, state_key, coder):
        if False:
            return 10
        'Materialized the first page of data, concatenated with a lazy iterable\n    of the rest, if any.\n    '
        (data, continuation_token) = self._underlying.get_raw(state_key, None)
        head = []
        input_stream = coder_impl.create_InputStream(data)
        while input_stream.size() > 0:
            head.append(coder.decode_from_stream(input_stream, True))
        if not continuation_token:
            return head
        else:
            return self.ContinuationIterable(head, functools.partial(self._lazy_iterator, state_key, coder, continuation_token))

    class ContinuationIterable(Generic[T], CacheAware):

        def __init__(self, head, continue_iterator_fn):
            if False:
                i = 10
                return i + 15
            self.head = head
            self.continue_iterator_fn = continue_iterator_fn

        def __iter__(self):
            if False:
                print('Hello World!')
            for item in self.head:
                yield item
            for item in self.continue_iterator_fn():
                yield item

        def get_referents_for_cache(self):
            if False:
                i = 10
                return i + 15
            return [self.head]

    @staticmethod
    def _convert_to_cache_key(state_key):
        if False:
            for i in range(10):
                print('nop')
        return state_key.SerializeToString()

class _Future(Generic[T]):
    """A simple future object to implement blocking requests.
  """

    def __init__(self):
        if False:
            return 10
        self._event = threading.Event()

    def wait(self, timeout=None):
        if False:
            print('Hello World!')
        return self._event.wait(timeout)

    def get(self, timeout=None):
        if False:
            return 10
        if self.wait(timeout):
            return self._value
        else:
            raise LookupError()

    def set(self, value):
        if False:
            return 10
        self._value = value
        self._event.set()
        return self

    @classmethod
    def done(cls):
        if False:
            while True:
                i = 10
        if not hasattr(cls, 'DONE'):
            done_future = _Future[None]()
            done_future.set(None)
            cls.DONE = done_future
        return cls.DONE

class _DeferredCall(_Future[T]):

    def __init__(self, func, *args):
        if False:
            return 10
        self._func = func
        self._args = [arg if isinstance(arg, _Future) else _Future().set(arg) for arg in args]

    def wait(self, timeout=None):
        if False:
            print('Hello World!')
        return all((arg.wait(timeout) for arg in self._args))

    def get(self, timeout=None):
        if False:
            print('Hello World!')
        return self._func(*(arg.get(timeout) for arg in self._args))

    def set(self, value):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

class KeyedDefaultDict(DefaultDict[_KT, _VT]):
    if TYPE_CHECKING:

        def __init__(self, default_factory):
            if False:
                i = 10
                return i + 15
            pass

    def __missing__(self, key):
        if False:
            return 10
        self[key] = self.default_factory(key)
        return self[key]