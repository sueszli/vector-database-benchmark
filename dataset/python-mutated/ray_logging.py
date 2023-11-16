import colorama
from dataclasses import dataclass
import logging
import os
import re
import sys
import threading
import time
from typing import Callable, Dict, List, Set, Tuple, Any, Optional
import ray
from ray.experimental.tqdm_ray import RAY_TQDM_MAGIC
from ray._private.ray_constants import RAY_DEDUP_LOGS, RAY_DEDUP_LOGS_AGG_WINDOW_S, RAY_DEDUP_LOGS_ALLOW_REGEX, RAY_DEDUP_LOGS_SKIP_REGEX
from ray.util.debug import log_once

def setup_logger(logging_level: int, logging_format: str):
    if False:
        print('Hello World!')
    'Setup default logging for ray.'
    logger = logging.getLogger('ray')
    if type(logging_level) is str:
        logging_level = logging.getLevelName(logging_level.upper())
    logger.setLevel(logging_level)

def setup_component_logger(*, logging_level, logging_format, log_dir, filename, max_bytes, backup_count, logger_name=None, propagate=True):
    if False:
        return 10
    'Configure the logger that is used for Ray\'s python components.\n\n    For example, it should be used for monitor, dashboard, and log monitor.\n    The only exception is workers. They use the different logging config.\n\n    Ray\'s python components generally should not write to stdout/stderr, because\n    messages written there will be redirected to the head node. For deployments where\n    there may be thousands of workers, this would create unacceptable levels of log\n    spam. For this reason, we disable the "ray" logger\'s handlers, and enable\n    propagation so that log messages that actually do need to be sent to the head node\n    can reach it.\n\n    Args:\n        logging_level: Logging level in string or logging enum.\n        logging_format: Logging format string.\n        log_dir: Log directory path. If empty, logs will go to\n            stderr.\n        filename: Name of the file to write logs. If empty, logs will go\n            to stderr.\n        max_bytes: Same argument as RotatingFileHandler\'s maxBytes.\n        backup_count: Same argument as RotatingFileHandler\'s backupCount.\n        logger_name: Used to create or get the correspoding\n            logger in getLogger call. It will get the root logger by default.\n        propagate: Whether to propagate the log to the parent logger.\n    Returns:\n        the created or modified logger.\n    '
    ray._private.log.clear_logger('ray')
    logger = logging.getLogger(logger_name)
    if type(logging_level) is str:
        logging_level = logging.getLevelName(logging_level.upper())
    if not filename or not log_dir:
        handler = logging.StreamHandler()
    else:
        handler = logging.handlers.RotatingFileHandler(os.path.join(log_dir, filename), maxBytes=max_bytes, backupCount=backup_count)
    handler.setLevel(logging_level)
    logger.setLevel(logging_level)
    handler.setFormatter(logging.Formatter(logging_format))
    logger.addHandler(handler)
    logger.propagate = propagate
    return logger

def run_callback_on_events_in_ipython(event: str, cb: Callable):
    if False:
        for i in range(10):
            print('nop')
    '\n    Register a callback to be run after each cell completes in IPython.\n    E.g.:\n        This is used to flush the logs after each cell completes.\n\n    If IPython is not installed, this function does nothing.\n\n    Args:\n        cb: The callback to run.\n    '
    if 'IPython' in sys.modules:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.events.register(event, cb)
'\nAll components underneath here is used specifically for the default_worker.py.\n'

def get_worker_log_file_name(worker_type, job_id=None):
    if False:
        return 10
    if job_id is None:
        job_id = os.environ.get('RAY_JOB_ID')
    if worker_type == 'WORKER':
        if job_id is None:
            job_id = ''
        worker_name = 'worker'
    else:
        job_id = ''
        worker_name = 'io_worker'
    assert ray._private.worker._global_node is not None
    assert ray._private.worker.global_worker is not None
    filename = f'{worker_name}-{ray.get_runtime_context().get_worker_id()}-'
    if job_id:
        filename += f'{job_id}-'
    filename += f'{os.getpid()}'
    return filename

def configure_log_file(out_file, err_file):
    if False:
        print('Hello World!')
    if out_file is None or err_file is None:
        return
    stdout_fileno = sys.stdout.fileno()
    stderr_fileno = sys.stderr.fileno()
    os.dup2(out_file.fileno(), stdout_fileno)
    os.dup2(err_file.fileno(), stderr_fileno)
    sys.stdout = ray._private.utils.open_log(stdout_fileno, unbuffered=True, closefd=False)
    sys.stderr = ray._private.utils.open_log(stderr_fileno, unbuffered=True, closefd=False)

class WorkerStandardStreamDispatcher:

    def __init__(self):
        if False:
            print('Hello World!')
        self.handlers = []
        self._lock = threading.Lock()

    def add_handler(self, name: str, handler: Callable) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            self.handlers.append((name, handler))

    def remove_handler(self, name: str) -> None:
        if False:
            print('Hello World!')
        with self._lock:
            new_handlers = [pair for pair in self.handlers if pair[0] != name]
            self.handlers = new_handlers

    def emit(self, data):
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            for pair in self.handlers:
                (_, handle) = pair
                handle(data)
global_worker_stdstream_dispatcher = WorkerStandardStreamDispatcher()
NUMBERS = re.compile('(\\d+|0x[0-9a-fA-F]+)')
LogBatch = Dict[str, Any]

def _canonicalise_log_line(line):
    if False:
        print('Hello World!')
    return ' '.join((x for x in line.split() if not NUMBERS.search(x)))

@dataclass
class DedupState:
    timestamp: int
    count: int
    line: int
    metadata: LogBatch
    sources: Set[Tuple[str, int]]

    def formatted(self) -> str:
        if False:
            print('Hello World!')
        return self.line + _color(f' [repeated {self.count}x across cluster]' + _warn_once())

class LogDeduplicator:

    def __init__(self, agg_window_s: int, allow_re: Optional[str], skip_re: Optional[str], *, _timesource=None):
        if False:
            print('Hello World!')
        self.agg_window_s = agg_window_s
        if allow_re:
            self.allow_re = re.compile(allow_re)
        else:
            self.allow_re = None
        if skip_re:
            self.skip_re = re.compile(skip_re)
        else:
            self.skip_re = None
        self.recent: Dict[str, DedupState] = {}
        self.timesource = _timesource or (lambda : time.time())
        run_callback_on_events_in_ipython('post_execute', self.flush)

    def deduplicate(self, batch: LogBatch) -> List[LogBatch]:
        if False:
            print('Hello World!')
        'Rewrite a batch of lines to reduce duplicate log messages.\n\n        Args:\n            batch: The batch of lines from a single source.\n\n        Returns:\n            List of batches from this and possibly other previous sources to print.\n        '
        if not RAY_DEDUP_LOGS:
            return [batch]
        now = self.timesource()
        metadata = batch.copy()
        del metadata['lines']
        source = (metadata.get('ip'), metadata.get('pid'))
        output: List[LogBatch] = [dict(**metadata, lines=[])]
        for line in batch['lines']:
            if RAY_TQDM_MAGIC in line or (self.allow_re and self.allow_re.search(line)):
                output[0]['lines'].append(line)
                continue
            elif self.skip_re and self.skip_re.search(line):
                continue
            dedup_key = _canonicalise_log_line(line)
            if dedup_key in self.recent:
                sources = self.recent[dedup_key].sources
                sources.add(source)
                if len(sources) > 1 or batch['pid'] == 'raylet':
                    state = self.recent[dedup_key]
                    self.recent[dedup_key] = DedupState(state.timestamp, state.count + 1, line, metadata, sources)
                else:
                    output[0]['lines'].append(line)
            else:
                self.recent[dedup_key] = DedupState(now, 0, line, metadata, {source})
                output[0]['lines'].append(line)
        while self.recent:
            if now - next(iter(self.recent.values())).timestamp < self.agg_window_s:
                break
            dedup_key = next(iter(self.recent))
            state = self.recent.pop(dedup_key)
            if state.count > 1:
                output.append(dict(**state.metadata, lines=[state.formatted()]))
                state.timestamp = now
                state.count = 0
                self.recent[dedup_key] = state
            elif state.count > 0:
                output.append(dict(state.metadata, lines=[state.line]))
        return output

    def flush(self) -> List[dict]:
        if False:
            i = 10
            return i + 15
        'Return all buffered log messages and clear the buffer.\n\n        Returns:\n            List of log batches to print.\n        '
        output = []
        for state in self.recent.values():
            if state.count > 1:
                output.append(dict(state.metadata, lines=[state.formatted()]))
            elif state.count > 0:
                output.append(dict(state.metadata, **{'lines': [state.line]}))
        self.recent.clear()
        return output

def _warn_once() -> str:
    if False:
        i = 10
        return i + 15
    if log_once('log_dedup_warning'):
        return ' (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/ray-logging.html#log-deduplication for more options.)'
    else:
        return ''

def _color(msg: str) -> str:
    if False:
        while True:
            i = 10
    return '{}{}{}'.format(colorama.Fore.GREEN, msg, colorama.Style.RESET_ALL)
stdout_deduplicator = LogDeduplicator(RAY_DEDUP_LOGS_AGG_WINDOW_S, RAY_DEDUP_LOGS_ALLOW_REGEX, RAY_DEDUP_LOGS_SKIP_REGEX)
stderr_deduplicator = LogDeduplicator(RAY_DEDUP_LOGS_AGG_WINDOW_S, RAY_DEDUP_LOGS_ALLOW_REGEX, RAY_DEDUP_LOGS_SKIP_REGEX)