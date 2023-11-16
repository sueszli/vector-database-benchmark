"""Worker status api handler for reporting SDK harness debug info."""
import logging
import queue
import sys
import threading
import time
import traceback
from collections import defaultdict
import grpc
from apache_beam.portability.api import beam_fn_api_pb2
from apache_beam.portability.api import beam_fn_api_pb2_grpc
from apache_beam.runners.worker.channel_factory import GRPCChannelFactory
from apache_beam.runners.worker.statecache import StateCache
from apache_beam.runners.worker.worker_id_interceptor import WorkerIdInterceptor
from apache_beam.utils.sentinel import Sentinel
try:
    from guppy import hpy
except ImportError:
    hpy = None
_LOGGER = logging.getLogger(__name__)
DEFAULT_LOG_LULL_TIMEOUT_NS = 5 * 60 * 1000 * 1000 * 1000
LOG_LULL_FULL_THREAD_DUMP_INTERVAL_S = 20 * 60
LOG_LULL_FULL_THREAD_DUMP_LULL_S = 20 * 60

def thread_dump():
    if False:
        print('Hello World!')
    'Get a thread dump for the current SDK worker harness. '
    stack_traces = defaultdict(list)
    frames = sys._current_frames()
    for t in threading.enumerate():
        try:
            stack_trace = ''.join(traceback.format_stack(frames[t.ident]))
        except KeyError:
            continue
        thread_ident_name = (t.ident, t.name)
        stack_traces[stack_trace].append(thread_ident_name)
    all_traces = ['=' * 10 + ' THREAD DUMP ' + '=' * 10]
    for (stack, identity) in stack_traces.items():
        (ident, name) = identity[0]
        trace = '--- Thread #%s name: %s %s---\n' % (ident, name, 'and other %d threads' % (len(identity) - 1) if len(identity) > 1 else '')
        if len(identity) > 1:
            trace += 'threads: %s\n' % identity
        trace += stack
        all_traces.append(trace)
    all_traces.append('=' * 30)
    return '\n'.join(all_traces)

def heap_dump():
    if False:
        print('Hello World!')
    'Get a heap dump for the current SDK worker harness. '
    banner = '=' * 10 + ' HEAP DUMP ' + '=' * 10 + '\n'
    if not hpy:
        heap = 'Unable to import guppy, the heap dump will be skipped.\n'
    else:
        heap = '%s\n' % hpy().heap()
    ending = '=' * 30
    return banner + heap + ending

def _state_cache_stats(state_cache):
    if False:
        while True:
            i = 10
    'Gather state cache statistics.'
    cache_stats = ['=' * 10 + ' CACHE STATS ' + '=' * 10]
    if not state_cache.is_cache_enabled():
        cache_stats.append('Cache disabled')
    else:
        cache_stats.append(state_cache.describe_stats())
    return '\n'.join(cache_stats)

def _active_processing_bundles_state(bundle_process_cache):
    if False:
        i = 10
        return i + 15
    'Gather information about the currently in-processing active bundles.\n\n  The result only keeps the longest lasting 10 bundles to avoid excessive\n  spamming.\n  '
    active_bundles = ['=' * 10 + ' ACTIVE PROCESSING BUNDLES ' + '=' * 10]
    if not bundle_process_cache.active_bundle_processors:
        active_bundles.append('No active processing bundles.')
    else:
        cache = []
        for instruction in list(bundle_process_cache.active_bundle_processors.keys()):
            processor = bundle_process_cache.lookup(instruction)
            if processor:
                info = processor.state_sampler.get_info()
                cache.append((instruction, processor.process_bundle_descriptor.id, info.tracked_thread, info.time_since_transition))
        cache.sort(key=lambda x: x[-1], reverse=True)
        for s in cache[:10]:
            state = '--- instruction %s ---\n' % s[0]
            state += 'ProcessBundleDescriptorId: %s\n' % s[1]
            state += 'tracked thread: %s\n' % s[2]
            state += 'time since transition: %.2f seconds\n' % (s[3] / 1000000000.0)
            active_bundles.append(state)
    active_bundles.append('=' * 30)
    return '\n'.join(active_bundles)
DONE = Sentinel.sentinel

class FnApiWorkerStatusHandler(object):
    """FnApiWorkerStatusHandler handles worker status request from Runner. """

    def __init__(self, status_address, bundle_process_cache=None, state_cache=None, enable_heap_dump=False, log_lull_timeout_ns=DEFAULT_LOG_LULL_TIMEOUT_NS):
        if False:
            i = 10
            return i + 15
        'Initialize FnApiWorkerStatusHandler.\n\n    Args:\n      status_address: The URL Runner uses to host the WorkerStatus server.\n      bundle_process_cache: The BundleProcessor cache dict from sdk worker.\n      state_cache: The StateCache form sdk worker.\n    '
        self._alive = True
        self._bundle_process_cache = bundle_process_cache
        self._state_cache = state_cache
        ch = GRPCChannelFactory.insecure_channel(status_address)
        grpc.channel_ready_future(ch).result(timeout=60)
        self._status_channel = grpc.intercept_channel(ch, WorkerIdInterceptor())
        self._status_stub = beam_fn_api_pb2_grpc.BeamFnWorkerStatusStub(self._status_channel)
        self._responses = queue.Queue()
        self.log_lull_timeout_ns = log_lull_timeout_ns
        self._last_full_thread_dump_secs = 0.0
        self._last_lull_logged_secs = 0.0
        self._server = threading.Thread(target=lambda : self._serve(), name='fn_api_status_handler')
        self._server.daemon = True
        self._enable_heap_dump = enable_heap_dump
        self._server.start()
        self._lull_logger = threading.Thread(target=lambda : self._log_lull_in_bundle_processor(self._bundle_process_cache), name='lull_operation_logger')
        self._lull_logger.daemon = True
        self._lull_logger.start()

    def _get_responses(self):
        if False:
            while True:
                i = 10
        while True:
            response = self._responses.get()
            if response is DONE:
                self._alive = False
                return
            yield response

    def _serve(self):
        if False:
            while True:
                i = 10
        while self._alive:
            for request in self._status_stub.WorkerStatus(self._get_responses()):
                try:
                    self._responses.put(beam_fn_api_pb2.WorkerStatusResponse(id=request.id, status_info=self.generate_status_response()))
                except Exception:
                    traceback_string = traceback.format_exc()
                    self._responses.put(beam_fn_api_pb2.WorkerStatusResponse(id=request.id, error='Exception encountered while generating status page: %s' % traceback_string))

    def generate_status_response(self):
        if False:
            while True:
                i = 10
        all_status_sections = []
        if self._state_cache:
            all_status_sections.append(_state_cache_stats(self._state_cache))
        if self._bundle_process_cache:
            all_status_sections.append(_active_processing_bundles_state(self._bundle_process_cache))
        all_status_sections.append(thread_dump())
        if self._enable_heap_dump:
            all_status_sections.append(heap_dump())
        return '\n'.join(all_status_sections)

    def close(self):
        if False:
            while True:
                i = 10
        self._responses.put(DONE, timeout=5)

    def _log_lull_in_bundle_processor(self, bundle_process_cache):
        if False:
            while True:
                i = 10
        while True:
            time.sleep(2 * 60)
            if bundle_process_cache and bundle_process_cache.active_bundle_processors:
                for instruction in list(bundle_process_cache.active_bundle_processors.keys()):
                    processor = bundle_process_cache.lookup(instruction)
                    if processor:
                        info = processor.state_sampler.get_info()
                        self._log_lull_sampler_info(info, instruction)

    def _log_lull_sampler_info(self, sampler_info, instruction):
        if False:
            print('Hello World!')
        if not self._passed_lull_timeout_since_last_log():
            return
        if sampler_info and sampler_info.time_since_transition and (sampler_info.time_since_transition > self.log_lull_timeout_ns):
            lull_seconds = sampler_info.time_since_transition / 1000000000.0
            step_name = sampler_info.state_name.step_name
            state_name = sampler_info.state_name.name
            if step_name and state_name:
                step_name_log = ' for PTransform{name=%s, state=%s}' % (step_name, state_name)
            else:
                step_name_log = ''
            stack_trace = self._get_stack_trace(sampler_info)
            _LOGGER.warning('Operation ongoing in bundle %s%s for at least %.2f seconds without outputting or completing.\nCurrent Traceback:\n%s', instruction, step_name_log, lull_seconds, stack_trace)

    def _get_stack_trace(self, sampler_info):
        if False:
            return 10
        exec_thread = getattr(sampler_info, 'tracked_thread', None)
        if exec_thread is not None:
            thread_frame = sys._current_frames().get(exec_thread.ident)
            return '\n'.join(traceback.format_stack(thread_frame)) if thread_frame else ''
        else:
            return '-NOT AVAILABLE-'

    def _passed_lull_timeout_since_last_log(self) -> bool:
        if False:
            while True:
                i = 10
        if time.time() - self._last_lull_logged_secs > self.log_lull_timeout_ns / 1000000000.0:
            self._last_lull_logged_secs = time.time()
            return True
        else:
            return False