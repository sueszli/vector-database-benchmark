"""Beam fn API log handler."""
import logging
import math
import queue
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Union
from typing import cast
import grpc
from apache_beam.portability.api import beam_fn_api_pb2
from apache_beam.portability.api import beam_fn_api_pb2_grpc
from apache_beam.runners.worker import statesampler
from apache_beam.runners.worker.channel_factory import GRPCChannelFactory
from apache_beam.runners.worker.worker_id_interceptor import WorkerIdInterceptor
from apache_beam.utils.sentinel import Sentinel
if TYPE_CHECKING:
    from apache_beam.portability.api import endpoints_pb2
LOG_LEVEL_TO_LOGENTRY_MAP = {logging.FATAL: beam_fn_api_pb2.LogEntry.Severity.CRITICAL, logging.ERROR: beam_fn_api_pb2.LogEntry.Severity.ERROR, logging.WARNING: beam_fn_api_pb2.LogEntry.Severity.WARN, logging.INFO: beam_fn_api_pb2.LogEntry.Severity.INFO, logging.DEBUG: beam_fn_api_pb2.LogEntry.Severity.DEBUG, logging.NOTSET: beam_fn_api_pb2.LogEntry.Severity.UNSPECIFIED, -float('inf'): beam_fn_api_pb2.LogEntry.Severity.DEBUG}
LOGENTRY_TO_LOG_LEVEL_MAP = {beam_fn_api_pb2.LogEntry.Severity.CRITICAL: logging.CRITICAL, beam_fn_api_pb2.LogEntry.Severity.ERROR: logging.ERROR, beam_fn_api_pb2.LogEntry.Severity.WARN: logging.WARNING, beam_fn_api_pb2.LogEntry.Severity.NOTICE: logging.INFO + 1, beam_fn_api_pb2.LogEntry.Severity.INFO: logging.INFO, beam_fn_api_pb2.LogEntry.Severity.DEBUG: logging.DEBUG, beam_fn_api_pb2.LogEntry.Severity.TRACE: logging.DEBUG - 1, beam_fn_api_pb2.LogEntry.Severity.UNSPECIFIED: logging.NOTSET}

class FnApiLogRecordHandler(logging.Handler):
    """A handler that writes log records to the fn API."""
    _MAX_BATCH_SIZE = 1000
    _FINISHED = Sentinel.sentinel
    _QUEUE_SIZE = 10000

    def __init__(self, log_service_descriptor):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._alive = True
        self._dropped_logs = 0
        self._log_entry_queue = queue.Queue(maxsize=self._QUEUE_SIZE)
        ch = GRPCChannelFactory.insecure_channel(log_service_descriptor.url)
        grpc.channel_ready_future(ch).result(timeout=60)
        self._log_channel = grpc.intercept_channel(ch, WorkerIdInterceptor())
        self._reader = threading.Thread(target=lambda : self._read_log_control_messages(), name='read_log_control_messages')
        self._reader.daemon = True
        self._reader.start()

    def connect(self):
        if False:
            print('Hello World!')
        if hasattr(self, '_logging_stub'):
            del self._logging_stub
        self._logging_stub = beam_fn_api_pb2_grpc.BeamFnLoggingStub(self._log_channel)
        return self._logging_stub.Logging(self._write_log_entries())

    def map_log_level(self, level):
        if False:
            i = 10
            return i + 15
        try:
            return LOG_LEVEL_TO_LOGENTRY_MAP[level]
        except KeyError:
            return max((beam_level for (python_level, beam_level) in LOG_LEVEL_TO_LOGENTRY_MAP.items() if python_level <= level))

    def emit(self, record):
        if False:
            for i in range(10):
                print('nop')
        log_entry = beam_fn_api_pb2.LogEntry()
        log_entry.severity = self.map_log_level(record.levelno)
        try:
            log_entry.message = self.format(record)
        except Exception:
            log_entry.message = "Failed to format '%s' with args '%s' during logging." % (str(record.msg), record.args)
        log_entry.thread = record.threadName
        log_entry.log_location = '%s:%s' % (record.pathname or record.module, record.lineno or record.funcName)
        (fraction, seconds) = math.modf(record.created)
        nanoseconds = 1000000000.0 * fraction
        log_entry.timestamp.seconds = int(seconds)
        log_entry.timestamp.nanos = int(nanoseconds)
        if record.exc_info:
            log_entry.trace = ''.join(traceback.format_exception(*record.exc_info))
        instruction_id = statesampler.get_current_instruction_id()
        if instruction_id:
            log_entry.instruction_id = instruction_id
        tracker = statesampler.get_current_tracker()
        if tracker:
            current_state = tracker.current_state()
            if current_state and current_state.name_context and current_state.name_context.transform_id:
                log_entry.transform_id = current_state.name_context.transform_id
        try:
            self._log_entry_queue.put(log_entry, block=False)
        except queue.Full:
            self._dropped_logs += 1

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        'Flush out all existing log entries and unregister this handler.'
        try:
            self._alive = False
            self.acquire()
            self._log_entry_queue.put(self._FINISHED, timeout=5)
            self._reader.join()
            self.release()
            super().close()
        except Exception:
            logging.error('Error closing the logging channel.', exc_info=True)

    def _write_log_entries(self):
        if False:
            print('Hello World!')
        done = False
        while not done:
            log_entries = [self._log_entry_queue.get()]
            try:
                for _ in range(self._MAX_BATCH_SIZE):
                    log_entries.append(self._log_entry_queue.get_nowait())
            except queue.Empty:
                pass
            if log_entries[-1] is self._FINISHED:
                done = True
                log_entries.pop()
            if log_entries:
                yield beam_fn_api_pb2.LogEntry.List(log_entries=cast(List[beam_fn_api_pb2.LogEntry], log_entries))

    def _read_log_control_messages(self):
        if False:
            return 10
        alive = True
        while alive:
            log_control_iterator = self.connect()
            if self._dropped_logs > 0:
                logging.warning('Dropped %d logs while logging client disconnected', self._dropped_logs)
                self._dropped_logs = 0
            try:
                for _ in log_control_iterator:
                    pass
                return
            except Exception as ex:
                print('Logging client failed: {}... resetting'.format(ex), file=sys.stderr)
                time.sleep(0.5)
            alive = self._alive