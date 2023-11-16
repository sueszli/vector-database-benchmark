"""Implements ProcessPoolExecutor.

The follow diagram and text describe the data-flow through the system:

|======================= In-process =====================|== Out-of-process ==|

+----------+     +----------+       +--------+     +-----------+    +---------+
|          |  => | Work Ids |       |        |     | Call Q    |    | Process |
|          |     +----------+       |        |     +-----------+    |  Pool   |
|          |     | ...      |       |        |     | ...       |    +---------+
|          |     | 6        |    => |        |  => | 5, call() | => |         |
|          |     | 7        |       |        |     | ...       |    |         |
| Process  |     | ...      |       | Local  |     +-----------+    | Process |
|  Pool    |     +----------+       | Worker |                      |  #1..n  |
| Executor |                        | Thread |                      |         |
|          |     +----------- +     |        |     +-----------+    |         |
|          | <=> | Work Items | <=> |        | <=  | Result Q  | <= |         |
|          |     +------------+     |        |     +-----------+    |         |
|          |     | 6: call()  |     |        |     | ...       |    |         |
|          |     |    future  |     +--------+     | 4, result |    |         |
|          |     | ...        |                    | 3, except |    |         |
+----------+     +------------+                    +-----------+    +---------+

Executor.submit() called:
- creates a uniquely numbered _WorkItem and adds it to the "Work Items" dict
- adds the id of the _WorkItem to the "Work Ids" queue

Local worker thread:
- reads work ids from the "Work Ids" queue and looks up the corresponding
  WorkItem from the "Work Items" dict: if the work item has been cancelled then
  it is simply removed from the dict, otherwise it is repackaged as a
  _CallItem and put in the "Call Q". New _CallItems are put in the "Call Q"
  until "Call Q" is full. NOTE: the size of the "Call Q" is kept small because
  calls placed in the "Call Q" can no longer be cancelled with Future.cancel().
- reads _ResultItems from "Result Q", updates the future stored in the
  "Work Items" dict and deletes the dict entry

Process #1..n:
- reads _CallItems from "Call Q", executes the calls, and puts the resulting
  _ResultItems in "Result Q"
"""
__author__ = 'Thomas Moreau (thomas.moreau.2010@gmail.com)'
import os
import gc
import sys
import queue
import struct
import weakref
import warnings
import itertools
import traceback
import threading
from time import time, sleep
import multiprocessing as mp
from functools import partial
from pickle import PicklingError
from concurrent.futures import Executor
from concurrent.futures._base import LOGGER
from concurrent.futures.process import BrokenProcessPool as _BPPException
from multiprocessing.connection import wait
from ._base import Future
from .backend import get_context
from .backend.context import cpu_count, _MAX_WINDOWS_WORKERS
from .backend.queues import Queue, SimpleQueue
from .backend.reduction import set_loky_pickler, get_loky_pickler_name
from .backend.utils import kill_process_tree, get_exitcodes_terminated_worker
from .initializers import _prepare_initializer
MAX_DEPTH = int(os.environ.get('LOKY_MAX_DEPTH', 10))
_CURRENT_DEPTH = 0
_MEMORY_LEAK_CHECK_DELAY = 1.0
_MAX_MEMORY_LEAK_SIZE = int(300000000.0)
try:
    from psutil import Process
    _USE_PSUTIL = True

    def _get_memory_usage(pid, force_gc=False):
        if False:
            print('Hello World!')
        if force_gc:
            gc.collect()
        mem_size = Process(pid).memory_info().rss
        mp.util.debug(f'psutil return memory size: {mem_size}')
        return mem_size
except ImportError:
    _USE_PSUTIL = False

class _ThreadWakeup:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._closed = False
        (self._reader, self._writer) = mp.Pipe(duplex=False)

    def close(self):
        if False:
            while True:
                i = 10
        if not self._closed:
            self._closed = True
            self._writer.close()
            self._reader.close()

    def wakeup(self):
        if False:
            print('Hello World!')
        if not self._closed:
            self._writer.send_bytes(b'')

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._closed:
            while self._reader.poll():
                self._reader.recv_bytes()

class _ExecutorFlags:
    """necessary references to maintain executor states without preventing gc

    It permits to keep the information needed by executor_manager_thread
    and crash_detection_thread to maintain the pool without preventing the
    garbage collection of unreferenced executors.
    """

    def __init__(self, shutdown_lock):
        if False:
            while True:
                i = 10
        self.shutdown = False
        self.broken = None
        self.kill_workers = False
        self.shutdown_lock = shutdown_lock

    def flag_as_shutting_down(self, kill_workers=None):
        if False:
            while True:
                i = 10
        with self.shutdown_lock:
            self.shutdown = True
            if kill_workers is not None:
                self.kill_workers = kill_workers

    def flag_as_broken(self, broken):
        if False:
            print('Hello World!')
        with self.shutdown_lock:
            self.shutdown = True
            self.broken = broken
_global_shutdown = False
_global_shutdown_lock = threading.Lock()
_threads_wakeups = weakref.WeakKeyDictionary()

def _python_exit():
    if False:
        print('Hello World!')
    global _global_shutdown
    _global_shutdown = True
    items = list(_threads_wakeups.items())
    if len(items) > 0:
        mp.util.debug(f'Interpreter shutting down. Waking up {{len(items)}}executor_manager_thread:\n{items}')
    for (_, (shutdown_lock, thread_wakeup)) in items:
        with shutdown_lock:
            thread_wakeup.wakeup()
    for (thread, _) in items:
        with _global_shutdown_lock:
            thread.join()
mp.util.register_after_fork(_threads_wakeups, lambda obj: obj.clear())
process_pool_executor_at_exit = None
EXTRA_QUEUED_CALLS = 1

class _RemoteTraceback(Exception):
    """Embed stringification of remote traceback in local traceback"""

    def __init__(self, tb=None):
        if False:
            return 10
        self.tb = f'\n"""\n{tb}"""'

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.tb

class _ExceptionWithTraceback:

    def __init__(self, exc):
        if False:
            i = 10
            return i + 15
        tb = getattr(exc, '__traceback__', None)
        if tb is None:
            (_, _, tb) = sys.exc_info()
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = ''.join(tb)
        self.exc = exc
        self.tb = tb

    def __reduce__(self):
        if False:
            print('Hello World!')
        return (_rebuild_exc, (self.exc, self.tb))

def _rebuild_exc(exc, tb):
    if False:
        for i in range(10):
            print('nop')
    exc.__cause__ = _RemoteTraceback(tb)
    return exc

class _WorkItem:
    __slots__ = ['future', 'fn', 'args', 'kwargs']

    def __init__(self, future, fn, args, kwargs):
        if False:
            i = 10
            return i + 15
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

class _ResultItem:

    def __init__(self, work_id, exception=None, result=None):
        if False:
            for i in range(10):
                print('nop')
        self.work_id = work_id
        self.exception = exception
        self.result = result

class _CallItem:

    def __init__(self, work_id, fn, args, kwargs):
        if False:
            i = 10
            return i + 15
        self.work_id = work_id
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.loky_pickler = get_loky_pickler_name()

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        set_loky_pickler(self.loky_pickler)
        return self.fn(*self.args, **self.kwargs)

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'CallItem({self.work_id}, {self.fn}, {self.args}, {self.kwargs})'

class _SafeQueue(Queue):
    """Safe Queue set exception to the future object linked to a job"""

    def __init__(self, max_size=0, ctx=None, pending_work_items=None, running_work_items=None, thread_wakeup=None, reducers=None):
        if False:
            print('Hello World!')
        self.thread_wakeup = thread_wakeup
        self.pending_work_items = pending_work_items
        self.running_work_items = running_work_items
        super().__init__(max_size, reducers=reducers, ctx=ctx)

    def _on_queue_feeder_error(self, e, obj):
        if False:
            print('Hello World!')
        if isinstance(obj, _CallItem):
            if isinstance(e, struct.error):
                raised_error = RuntimeError('The task could not be sent to the workers as it is too large for `send_bytes`.')
            else:
                raised_error = PicklingError('Could not pickle the task to send it to the workers.')
            tb = traceback.format_exception(type(e), e, getattr(e, '__traceback__', None))
            raised_error.__cause__ = _RemoteTraceback(''.join(tb))
            work_item = self.pending_work_items.pop(obj.work_id, None)
            self.running_work_items.remove(obj.work_id)
            if work_item is not None:
                work_item.future.set_exception(raised_error)
                del work_item
            self.thread_wakeup.wakeup()
        else:
            super()._on_queue_feeder_error(e, obj)

def _get_chunks(chunksize, *iterables):
    if False:
        while True:
            i = 10
    'Iterates over zip()ed iterables in chunks.'
    it = zip(*iterables)
    while True:
        chunk = tuple(itertools.islice(it, chunksize))
        if not chunk:
            return
        yield chunk

def _process_chunk(fn, chunk):
    if False:
        i = 10
        return i + 15
    'Processes a chunk of an iterable passed to map.\n\n    Runs the function passed to map() on a chunk of the\n    iterable passed to map.\n\n    This function is run in a separate process.\n\n    '
    return [fn(*args) for args in chunk]

def _sendback_result(result_queue, work_id, result=None, exception=None):
    if False:
        i = 10
        return i + 15
    'Safely send back the given result or exception'
    try:
        result_queue.put(_ResultItem(work_id, result=result, exception=exception))
    except BaseException as e:
        exc = _ExceptionWithTraceback(e)
        result_queue.put(_ResultItem(work_id, exception=exc))

def _process_worker(call_queue, result_queue, initializer, initargs, processes_management_lock, timeout, worker_exit_lock, current_depth):
    if False:
        print('Hello World!')
    'Evaluates calls from call_queue and places the results in result_queue.\n\n    This worker is run in a separate process.\n\n    Args:\n        call_queue: A ctx.Queue of _CallItems that will be read and\n            evaluated by the worker.\n        result_queue: A ctx.Queue of _ResultItems that will written\n            to by the worker.\n        initializer: A callable initializer, or None\n        initargs: A tuple of args for the initializer\n        processes_management_lock: A ctx.Lock avoiding worker timeout while\n            some workers are being spawned.\n        timeout: maximum time to wait for a new item in the call_queue. If that\n            time is expired, the worker will shutdown.\n        worker_exit_lock: Lock to avoid flagging the executor as broken on\n            workers timeout.\n        current_depth: Nested parallelism level, to avoid infinite spawning.\n    '
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            LOGGER.critical('Exception in initializer:', exc_info=True)
            return
    global _CURRENT_DEPTH
    _CURRENT_DEPTH = current_depth
    _process_reference_size = None
    _last_memory_leak_check = None
    pid = os.getpid()
    mp.util.debug(f'Worker started with timeout={timeout}')
    while True:
        try:
            call_item = call_queue.get(block=True, timeout=timeout)
            if call_item is None:
                mp.util.info('Shutting down worker on sentinel')
        except queue.Empty:
            mp.util.info(f'Shutting down worker after timeout {timeout:0.3f}s')
            if processes_management_lock.acquire(block=False):
                processes_management_lock.release()
                call_item = None
            else:
                mp.util.info('Could not acquire processes_management_lock')
                continue
        except BaseException:
            previous_tb = traceback.format_exc()
            try:
                result_queue.put(_RemoteTraceback(previous_tb))
            except BaseException:
                print(previous_tb)
            mp.util.debug('Exiting with code 1')
            sys.exit(1)
        if call_item is None:
            result_queue.put(pid)
            is_clean = worker_exit_lock.acquire(True, timeout=30)
            _python_exit()
            if is_clean:
                mp.util.debug('Exited cleanly')
            else:
                mp.util.info('Main process did not release worker_exit')
            return
        try:
            r = call_item()
        except BaseException as e:
            exc = _ExceptionWithTraceback(e)
            result_queue.put(_ResultItem(call_item.work_id, exception=exc))
        else:
            _sendback_result(result_queue, call_item.work_id, result=r)
            del r
        del call_item
        if _USE_PSUTIL:
            if _process_reference_size is None:
                _process_reference_size = _get_memory_usage(pid, force_gc=True)
                _last_memory_leak_check = time()
                continue
            if time() - _last_memory_leak_check > _MEMORY_LEAK_CHECK_DELAY:
                mem_usage = _get_memory_usage(pid)
                _last_memory_leak_check = time()
                if mem_usage - _process_reference_size < _MAX_MEMORY_LEAK_SIZE:
                    continue
                mem_usage = _get_memory_usage(pid, force_gc=True)
                _last_memory_leak_check = time()
                if mem_usage - _process_reference_size < _MAX_MEMORY_LEAK_SIZE:
                    continue
                mp.util.info('Memory leak detected: shutting down worker')
                result_queue.put(pid)
                with worker_exit_lock:
                    mp.util.debug('Exit due to memory leak')
                    return
        elif _last_memory_leak_check is None or time() - _last_memory_leak_check > _MEMORY_LEAK_CHECK_DELAY:
            gc.collect()
            _last_memory_leak_check = time()

class _ExecutorManagerThread(threading.Thread):
    """Manages the communication between this process and the worker processes.

    The manager is run in a local thread.

    Args:
        executor: A reference to the ProcessPoolExecutor that owns
            this thread. A weakref will be own by the manager as well as
            references to internal objects used to introspect the state of
            the executor.
    """

    def __init__(self, executor):
        if False:
            print('Hello World!')
        self.thread_wakeup = executor._executor_manager_thread_wakeup
        self.shutdown_lock = executor._shutdown_lock

        def weakref_cb(_, thread_wakeup=self.thread_wakeup, shutdown_lock=self.shutdown_lock):
            if False:
                return 10
            if mp is not None:
                mp.util.debug('Executor collected: triggering callback for QueueManager wakeup')
            with shutdown_lock:
                thread_wakeup.wakeup()
        self.executor_reference = weakref.ref(executor, weakref_cb)
        self.executor_flags = executor._flags
        self.processes = executor._processes
        self.call_queue = executor._call_queue
        self.result_queue = executor._result_queue
        self.work_ids_queue = executor._work_ids
        self.pending_work_items = executor._pending_work_items
        self.running_work_items = executor._running_work_items
        self.processes_management_lock = executor._processes_management_lock
        super().__init__(name='ExecutorManagerThread')
        if sys.version_info < (3, 9):
            self.daemon = True

    def run(self):
        if False:
            return 10
        while True:
            self.add_call_item_to_queue()
            (result_item, is_broken, bpe) = self.wait_result_broken_or_wakeup()
            if is_broken:
                self.terminate_broken(bpe)
                return
            if result_item is not None:
                self.process_result_item(result_item)
                del result_item
            if self.is_shutting_down():
                self.flag_executor_shutting_down()
                if not self.pending_work_items:
                    self.join_executor_internals()
                    return

    def add_call_item_to_queue(self):
        if False:
            for i in range(10):
                print('nop')
        while True:
            if self.call_queue.full():
                return
            try:
                work_id = self.work_ids_queue.get(block=False)
            except queue.Empty:
                return
            else:
                work_item = self.pending_work_items[work_id]
                if work_item.future.set_running_or_notify_cancel():
                    self.running_work_items += [work_id]
                    self.call_queue.put(_CallItem(work_id, work_item.fn, work_item.args, work_item.kwargs), block=True)
                else:
                    del self.pending_work_items[work_id]
                    continue

    def wait_result_broken_or_wakeup(self):
        if False:
            i = 10
            return i + 15
        result_reader = self.result_queue._reader
        wakeup_reader = self.thread_wakeup._reader
        readers = [result_reader, wakeup_reader]
        worker_sentinels = [p.sentinel for p in list(self.processes.values())]
        ready = wait(readers + worker_sentinels)
        bpe = None
        is_broken = True
        result_item = None
        if result_reader in ready:
            try:
                result_item = result_reader.recv()
                if isinstance(result_item, _RemoteTraceback):
                    bpe = BrokenProcessPool('A task has failed to un-serialize. Please ensure that the arguments of the function are all picklable.')
                    bpe.__cause__ = result_item
                else:
                    is_broken = False
            except BaseException as e:
                bpe = BrokenProcessPool('A result has failed to un-serialize. Please ensure that the objects returned by the function are always picklable.')
                tb = traceback.format_exception(type(e), e, getattr(e, '__traceback__', None))
                bpe.__cause__ = _RemoteTraceback(''.join(tb))
        elif wakeup_reader in ready:
            is_broken = False
        else:
            exit_codes = ''
            if sys.platform != 'win32':
                exit_codes = f'\nThe exit codes of the workers are {get_exitcodes_terminated_worker(self.processes)}'
            mp.util.debug('A worker unexpectedly terminated. Workers that might have caused the breakage: ' + str({p.name: p.exitcode for p in list(self.processes.values()) if p is not None and p.sentinel in ready}))
            bpe = TerminatedWorkerError(f'A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker.\n{exit_codes}')
        self.thread_wakeup.clear()
        return (result_item, is_broken, bpe)

    def process_result_item(self, result_item):
        if False:
            while True:
                i = 10
        if isinstance(result_item, int):
            with self.processes_management_lock:
                p = self.processes.pop(result_item, None)
            if p is not None:
                p._worker_exit_lock.release()
                mp.util.debug(f'joining {p.name} when processing {p.pid} as result_item')
                p.join()
                del p
            n_pending = len(self.pending_work_items)
            n_running = len(self.running_work_items)
            if n_pending - n_running > 0 or n_running > len(self.processes):
                executor = self.executor_reference()
                if executor is not None and len(self.processes) < executor._max_workers:
                    warnings.warn('A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.', UserWarning)
                    with executor._processes_management_lock:
                        executor._adjust_process_count()
                    executor = None
        else:
            work_item = self.pending_work_items.pop(result_item.work_id, None)
            if work_item is not None:
                if result_item.exception:
                    work_item.future.set_exception(result_item.exception)
                else:
                    work_item.future.set_result(result_item.result)
                self.running_work_items.remove(result_item.work_id)

    def is_shutting_down(self):
        if False:
            for i in range(10):
                print('nop')
        executor = self.executor_reference()
        return _global_shutdown or ((executor is None or self.executor_flags.shutdown) and (not self.executor_flags.broken))

    def terminate_broken(self, bpe):
        if False:
            while True:
                i = 10
        self.executor_flags.flag_as_broken(bpe)
        for work_item in self.pending_work_items.values():
            work_item.future.set_exception(bpe)
            del work_item
        self.pending_work_items.clear()
        self.kill_workers(reason='broken executor')
        self.join_executor_internals()

    def flag_executor_shutting_down(self):
        if False:
            return 10
        self.executor_flags.flag_as_shutting_down()
        if self.executor_flags.kill_workers:
            while self.pending_work_items:
                (_, work_item) = self.pending_work_items.popitem()
                work_item.future.set_exception(ShutdownExecutorError('The Executor was shutdown with `kill_workers=True` before this job could complete.'))
                del work_item
            self.kill_workers(reason='executor shutting down')

    def kill_workers(self, reason=''):
        if False:
            return 10
        while self.processes:
            (_, p) = self.processes.popitem()
            mp.util.debug(f'terminate process {p.name}, reason: {reason}')
            try:
                kill_process_tree(p)
            except ProcessLookupError:
                pass

    def shutdown_workers(self):
        if False:
            for i in range(10):
                print('nop')
        with self.processes_management_lock:
            n_children_to_stop = 0
            for p in list(self.processes.values()):
                mp.util.debug(f'releasing worker exit lock on {p.name}')
                p._worker_exit_lock.release()
                n_children_to_stop += 1
        mp.util.debug(f'found {n_children_to_stop} processes to stop')
        n_sentinels_sent = 0
        cooldown_time = 0.001
        while n_sentinels_sent < n_children_to_stop and self.get_n_children_alive() > 0:
            for _ in range(n_children_to_stop - n_sentinels_sent):
                try:
                    self.call_queue.put_nowait(None)
                    n_sentinels_sent += 1
                except queue.Full as e:
                    if cooldown_time > 5.0:
                        mp.util.info(f'failed to send all sentinels and exit with error.\ncall_queue size={self.call_queue._maxsize};  full is {self.call_queue.full()}; ')
                        raise e
                    mp.util.info('full call_queue prevented to send all sentinels at once, waiting...')
                    sleep(cooldown_time)
                    cooldown_time *= 1.2
                    break
        mp.util.debug(f'sent {n_sentinels_sent} sentinels to the call queue')

    def join_executor_internals(self):
        if False:
            print('Hello World!')
        self.shutdown_workers()
        mp.util.debug('closing call_queue')
        self.call_queue.close()
        self.call_queue.join_thread()
        mp.util.debug('closing result_queue')
        self.result_queue.close()
        mp.util.debug('closing thread_wakeup')
        with self.shutdown_lock:
            self.thread_wakeup.close()
        with self.processes_management_lock:
            mp.util.debug(f'joining {len(self.processes)} processes')
            n_joined_processes = 0
            while True:
                try:
                    (pid, p) = self.processes.popitem()
                    mp.util.debug(f'joining process {p.name} with pid {pid}')
                    p.join()
                    n_joined_processes += 1
                except KeyError:
                    break
            mp.util.debug(f'executor management thread clean shutdown of {n_joined_processes} workers')

    def get_n_children_alive(self):
        if False:
            i = 10
            return i + 15
        with self.processes_management_lock:
            return sum((p.is_alive() for p in list(self.processes.values())))
_system_limits_checked = False
_system_limited = None

def _check_system_limits():
    if False:
        for i in range(10):
            print('nop')
    global _system_limits_checked, _system_limited
    if _system_limits_checked and _system_limited:
        raise NotImplementedError(_system_limited)
    _system_limits_checked = True
    try:
        nsems_max = os.sysconf('SC_SEM_NSEMS_MAX')
    except (AttributeError, ValueError):
        return
    if nsems_max == -1:
        return
    if nsems_max >= 256:
        return
    _system_limited = f'system provides too few semaphores ({nsems_max} available, 256 necessary)'
    raise NotImplementedError(_system_limited)

def _chain_from_iterable_of_lists(iterable):
    if False:
        print('Hello World!')
    '\n    Specialized implementation of itertools.chain.from_iterable.\n    Each item in *iterable* should be a list.  This function is\n    careful not to keep references to yielded objects.\n    '
    for element in iterable:
        element.reverse()
        while element:
            yield element.pop()

def _check_max_depth(context):
    if False:
        i = 10
        return i + 15
    global _CURRENT_DEPTH
    if context.get_start_method() == 'fork' and _CURRENT_DEPTH > 0:
        raise LokyRecursionError("Could not spawn extra nested processes at depth superior to MAX_DEPTH=1. It is not possible to increase this limit when using the 'fork' start method.")
    if 0 < MAX_DEPTH and _CURRENT_DEPTH + 1 > MAX_DEPTH:
        raise LokyRecursionError(f'Could not spawn extra nested processes at depth superior to MAX_DEPTH={MAX_DEPTH}. If this is intendend, you can change this limit with the LOKY_MAX_DEPTH environment variable.')

class LokyRecursionError(RuntimeError):
    """A process tries to spawn too many levels of nested processes."""

class BrokenProcessPool(_BPPException):
    """
    Raised when the executor is broken while a future was in the running state.
    The cause can an error raised when unpickling the task in the worker
    process or when unpickling the result value in the parent process. It can
    also be caused by a worker process being terminated unexpectedly.
    """

class TerminatedWorkerError(BrokenProcessPool):
    """
    Raised when a process in a ProcessPoolExecutor terminated abruptly
    while a future was in the running state.
    """
BrokenExecutor = BrokenProcessPool

class ShutdownExecutorError(RuntimeError):
    """
    Raised when a ProcessPoolExecutor is shutdown while a future was in the
    running or pending state.
    """

class ProcessPoolExecutor(Executor):
    _at_exit = None

    def __init__(self, max_workers=None, job_reducers=None, result_reducers=None, timeout=None, context=None, initializer=None, initargs=(), env=None):
        if False:
            print('Hello World!')
        'Initializes a new ProcessPoolExecutor instance.\n\n        Args:\n            max_workers: int, optional (default: cpu_count())\n                The maximum number of processes that can be used to execute the\n                given calls. If None or not given then as many worker processes\n                will be created as the number of CPUs the current process\n                can use.\n            job_reducers, result_reducers: dict(type: reducer_func)\n                Custom reducer for pickling the jobs and the results from the\n                Executor. If only `job_reducers` is provided, `result_reducer`\n                will use the same reducers\n            timeout: int, optional (default: None)\n                Idle workers exit after timeout seconds. If a new job is\n                submitted after the timeout, the executor will start enough\n                new Python processes to make sure the pool of workers is full.\n            context: A multiprocessing context to launch the workers. This\n                object should provide SimpleQueue, Queue and Process.\n            initializer: An callable used to initialize worker processes.\n            initargs: A tuple of arguments to pass to the initializer.\n            env: A dict of environment variable to overwrite in the child\n                process. The environment variables are set before any module is\n                loaded. Note that this only works with the loky context.\n        '
        _check_system_limits()
        if max_workers is None:
            self._max_workers = cpu_count()
        else:
            if max_workers <= 0:
                raise ValueError('max_workers must be greater than 0')
            self._max_workers = max_workers
        if sys.platform == 'win32' and self._max_workers > _MAX_WINDOWS_WORKERS:
            warnings.warn(f'On Windows, max_workers cannot exceed {_MAX_WINDOWS_WORKERS} due to limitations of the operating system.')
            self._max_workers = _MAX_WINDOWS_WORKERS
        if context is None:
            context = get_context()
        self._context = context
        self._env = env
        (self._initializer, self._initargs) = _prepare_initializer(initializer, initargs)
        _check_max_depth(self._context)
        if result_reducers is None:
            result_reducers = job_reducers
        self._timeout = timeout
        self._executor_manager_thread = None
        self._processes = {}
        self._processes = {}
        self._queue_count = 0
        self._pending_work_items = {}
        self._running_work_items = []
        self._work_ids = queue.Queue()
        self._processes_management_lock = self._context.Lock()
        self._executor_manager_thread = None
        self._shutdown_lock = threading.Lock()
        self._executor_manager_thread_wakeup = _ThreadWakeup()
        self._flags = _ExecutorFlags(self._shutdown_lock)
        self._setup_queues(job_reducers, result_reducers)
        mp.util.debug('ProcessPoolExecutor is setup')

    def _setup_queues(self, job_reducers, result_reducers, queue_size=None):
        if False:
            print('Hello World!')
        if queue_size is None:
            queue_size = 2 * self._max_workers + EXTRA_QUEUED_CALLS
        self._call_queue = _SafeQueue(max_size=queue_size, pending_work_items=self._pending_work_items, running_work_items=self._running_work_items, thread_wakeup=self._executor_manager_thread_wakeup, reducers=job_reducers, ctx=self._context)
        self._call_queue._ignore_epipe = True
        self._result_queue = SimpleQueue(reducers=result_reducers, ctx=self._context)

    def _start_executor_manager_thread(self):
        if False:
            i = 10
            return i + 15
        if self._executor_manager_thread is None:
            mp.util.debug('_start_executor_manager_thread called')
            self._executor_manager_thread = _ExecutorManagerThread(self)
            self._executor_manager_thread.start()
            _threads_wakeups[self._executor_manager_thread] = (self._shutdown_lock, self._executor_manager_thread_wakeup)
            global process_pool_executor_at_exit
            if process_pool_executor_at_exit is None:
                if sys.version_info < (3, 9):
                    process_pool_executor_at_exit = mp.util.Finalize(None, _python_exit, exitpriority=20)
                else:
                    process_pool_executor_at_exit = threading._register_atexit(_python_exit)

    def _adjust_process_count(self):
        if False:
            while True:
                i = 10
        while len(self._processes) < self._max_workers:
            worker_exit_lock = self._context.BoundedSemaphore(1)
            args = (self._call_queue, self._result_queue, self._initializer, self._initargs, self._processes_management_lock, self._timeout, worker_exit_lock, _CURRENT_DEPTH + 1)
            worker_exit_lock.acquire()
            try:
                p = self._context.Process(target=_process_worker, args=args, env=self._env)
            except TypeError:
                p = self._context.Process(target=_process_worker, args=args)
            p._worker_exit_lock = worker_exit_lock
            p.start()
            self._processes[p.pid] = p
        mp.util.debug(f'Adjusted process count to {self._max_workers}: {[(p.name, pid) for (pid, p) in self._processes.items()]}')

    def _ensure_executor_running(self):
        if False:
            print('Hello World!')
        'ensures all workers and management thread are running'
        with self._processes_management_lock:
            if len(self._processes) != self._max_workers:
                self._adjust_process_count()
            self._start_executor_manager_thread()

    def submit(self, fn, *args, **kwargs):
        if False:
            return 10
        with self._flags.shutdown_lock:
            if self._flags.broken is not None:
                raise self._flags.broken
            if self._flags.shutdown:
                raise ShutdownExecutorError('cannot schedule new futures after shutdown')
            if _global_shutdown:
                raise RuntimeError('cannot schedule new futures after interpreter shutdown')
            f = Future()
            w = _WorkItem(f, fn, args, kwargs)
            self._pending_work_items[self._queue_count] = w
            self._work_ids.put(self._queue_count)
            self._queue_count += 1
            self._executor_manager_thread_wakeup.wakeup()
            self._ensure_executor_running()
            return f
    submit.__doc__ = Executor.submit.__doc__

    def map(self, fn, *iterables, **kwargs):
        if False:
            print('Hello World!')
        'Returns an iterator equivalent to map(fn, iter).\n\n        Args:\n            fn: A callable that will take as many arguments as there are\n                passed iterables.\n            timeout: The maximum number of seconds to wait. If None, then there\n                is no limit on the wait time.\n            chunksize: If greater than one, the iterables will be chopped into\n                chunks of size chunksize and submitted to the process pool.\n                If set to one, the items in the list will be sent one at a\n                time.\n\n        Returns:\n            An iterator equivalent to: map(func, *iterables) but the calls may\n            be evaluated out-of-order.\n\n        Raises:\n            TimeoutError: If the entire result iterator could not be generated\n                before the given timeout.\n            Exception: If fn(*args) raises for any values.\n        '
        timeout = kwargs.get('timeout', None)
        chunksize = kwargs.get('chunksize', 1)
        if chunksize < 1:
            raise ValueError('chunksize must be >= 1.')
        results = super().map(partial(_process_chunk, fn), _get_chunks(chunksize, *iterables), timeout=timeout)
        return _chain_from_iterable_of_lists(results)

    def shutdown(self, wait=True, kill_workers=False):
        if False:
            return 10
        mp.util.debug(f'shutting down executor {self}')
        self._flags.flag_as_shutting_down(kill_workers)
        executor_manager_thread = self._executor_manager_thread
        executor_manager_thread_wakeup = self._executor_manager_thread_wakeup
        if executor_manager_thread_wakeup is not None:
            with self._shutdown_lock:
                self._executor_manager_thread_wakeup.wakeup()
        if executor_manager_thread is not None and wait:
            with _global_shutdown_lock:
                executor_manager_thread.join()
                _threads_wakeups.pop(executor_manager_thread, None)
        self._executor_manager_thread = None
        self._executor_manager_thread_wakeup = None
        self._call_queue = None
        self._result_queue = None
        self._processes_management_lock = None
    shutdown.__doc__ = Executor.shutdown.__doc__