"""Module for `ClusterCoordinator` and relevant cluster-worker related library.

This is currently under development and the API is subject to change.
"""
import collections
import contextlib
import os
import re
import threading
import time
import weakref
from six.moves import queue
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.distribute.coordinator import values as values_lib
from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
_WORKER_MAXIMUM_RECOVERY_SEC = 3600
_POLL_FREQ_IN_SEC = 5
_CLOSURE_QUEUE_MAX_SIZE = 256 * 1024
_RPC_ERROR_FROM_PS = 'GRPC error information from remote target /job:ps'
_JOB_WORKER_STRING_IDENTIFIER = '/job:worker'
RemoteValueStatus = remote_value.RemoteValueStatus
RemoteValue = remote_value.RemoteValue
RemoteValueImpl = values_lib.RemoteValueImpl
RemoteVariable = values_lib.RemoteVariable
PerWorkerValues = values_lib.PerWorkerValues

class ClosureInputError(Exception):
    """Wrapper for errors from resource building.

  When a closure starts, it first checks for errors in any of its inputs, which
  are RemoteValues from resource closures. If there were any errors, it wraps
  the exception in this class and raises so it can be handled by the worker
  failure handler.

  Attributes:
    original_exception:
  """

    def __init__(self, original_exception):
        if False:
            while True:
                i = 10
        if isinstance(original_exception, (ClosureInputError, ClosureAbortedError)):
            self.original_exception = original_exception.original_exception
        else:
            self.original_exception = original_exception
        message = 'Input has an error, the original exception is %r, error message is %s.' % (self.original_exception, str(self.original_exception))
        super().__init__(message)
        self.with_traceback(original_exception.__traceback__)

class ClosureAbortedError(Exception):
    """Wrapper for errors from training closures, to attach to resource closures.

  This wrapper is used when a dependent training closure fails to set errors on
  its required resource closures.

  Attributes:
    original_exception: The Exception to wrap
  """

    def __init__(self, original_exception):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(original_exception, (ClosureInputError, ClosureAbortedError)):
            self.original_exception = original_exception.original_exception
        else:
            self.original_exception = original_exception
        message = 'Other function has an execution error, as a result, the current value is not available. The original exception is %r, error message is %s.' % (self.original_exception, str(self.original_exception))
        super().__init__(message)
        self.with_traceback(original_exception.__traceback__)

class PSUnavailableError(errors.UnavailableError):
    """Specifies that a parameter server is the unavailable task."""

    def __init__(self, original_exception):
        if False:
            while True:
                i = 10
        assert isinstance(original_exception, errors.UnavailableError)
        self.original_exception = original_exception
        super().__init__(original_exception.node_def, original_exception.op, original_exception.message)

def _get_error_from_remote_values(structure):
    if False:
        while True:
            i = 10
    'Attempts to return errors from `RemoteValue`s. Rebuilds them if needed.'
    errors_in_structure = []

    def _get_error(val):
        if False:
            while True:
                i = 10
        if isinstance(val, RemoteValue):
            error = val._get_error()
            if error:
                errors_in_structure.append(error)
    nest.map_structure(_get_error, structure)
    if errors_in_structure:
        return errors_in_structure[0]
    else:
        return None

def _maybe_as_type_spec(val):
    if False:
        while True:
            i = 10
    if isinstance(val, (RemoteValue, PerWorkerValues)):
        if val._type_spec is None:
            raise ValueError('Output of a scheduled function that is not tf.function cannot be the input of another function.')
        return val._type_spec
    else:
        return val

def _select_worker_slice(worker_id, structured):
    if False:
        while True:
            i = 10
    'Selects the worker slice of each of the items in `structured`.'

    def _get(x):
        if False:
            for i in range(10):
                print('nop')
        return x._values[worker_id] if isinstance(x, PerWorkerValues) else x
    return nest.map_structure(_get, structured)

def _disallow_remote_value_as_input(structured):
    if False:
        i = 10
        return i + 15
    'Raises if any element of `structured` is a RemoteValue.'

    def _raise_if_remote_value(x):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(x, RemoteValue):
            raise ValueError('`tf.distribute.experimental.coordinator.RemoteValue` used as an input to scheduled function is not yet supported.')
    nest.map_structure(_raise_if_remote_value, structured)

class Closure(object):
    """Hold a function to be scheduled and its arguments."""

    def __init__(self, function, cancellation_mgr, args=None, kwargs=None):
        if False:
            return 10
        if not callable(function):
            raise ValueError('Function passed to `ClusterCoordinator.schedule` must be a callable object.')
        self._args = args or ()
        self._kwargs = kwargs or {}
        _disallow_remote_value_as_input(self._args)
        _disallow_remote_value_as_input(self._kwargs)
        if isinstance(function, def_function.Function):
            replica_args = _select_worker_slice(0, self._args)
            replica_kwargs = _select_worker_slice(0, self._kwargs)
            with metric_utils.monitored_timer('function_tracing', state_tracker=function._get_tracing_count):
                self._concrete_function = function.get_concrete_function(*nest.map_structure(_maybe_as_type_spec, replica_args), **nest.map_structure(_maybe_as_type_spec, replica_kwargs))
        elif isinstance(function, tf_function.ConcreteFunction):
            self._concrete_function = function
        if hasattr(self, '_concrete_function'):
            self._output_type_spec = func_graph.convert_structure_to_signature(self._concrete_function.structured_outputs)
            self._function = cancellation_mgr.get_cancelable_function(self._concrete_function)
        else:
            self._output_type_spec = None
            self._function = function
        self._output_remote_value_ref = None

    def build_output_remote_value(self):
        if False:
            print('Hello World!')
        if self._output_remote_value_ref is None:
            ret = RemoteValueImpl(None, self._output_type_spec)
            self._output_remote_value_ref = weakref.ref(ret)
            return ret
        else:
            raise ValueError('The output of the Closure cannot be built more than once.')

    def maybe_call_with_output_remote_value(self, method):
        if False:
            i = 10
            return i + 15
        if self._output_remote_value_ref is None:
            return None
        output_remote_value = self._output_remote_value_ref()
        if output_remote_value is not None:
            return method(output_remote_value)
        return None

    def mark_cancelled(self):
        if False:
            while True:
                i = 10
        e = errors.CancelledError(None, None, 'The corresponding function is cancelled. Please reschedule the function.')
        self.maybe_call_with_output_remote_value(lambda r: r._set_error(e))

    def execute_on(self, worker):
        if False:
            return 10
        'Executes the closure on the given worker.\n\n    Args:\n      worker: a `Worker` object.\n    '
        replica_args = _select_worker_slice(worker.worker_index, self._args)
        replica_kwargs = _select_worker_slice(worker.worker_index, self._kwargs)
        e = _get_error_from_remote_values(replica_args) or _get_error_from_remote_values(replica_kwargs)
        if e:
            if not isinstance(e, ClosureInputError):
                e = ClosureInputError(e)
            raise e
        with ops.device(worker.device_name):
            with context.executor_scope(worker.executor):
                with coordinator_context.with_dispatch_context(worker):
                    with metric_utils.monitored_timer('closure_execution'):
                        output_values = self._function(*nest.map_structure(coordinator_context.maybe_get_remote_value, replica_args), **nest.map_structure(coordinator_context.maybe_get_remote_value, replica_kwargs))
        self.maybe_call_with_output_remote_value(lambda r: r._set_values(output_values))

class ResourceClosure(Closure):
    """A closure that builds a resource on a worker.

  ResourceClosures keep a reference to the closure object, which is used to
  rerun the closure upon recovery to ensure  workers have access to the
  resources they need.
  """

    def _init_remote_value(self):
        if False:
            print('Hello World!')
        return RemoteValueImpl(self, self._output_type_spec)

    def build_output_remote_value(self):
        if False:
            while True:
                i = 10
        if self._output_remote_value_ref is None:
            ret = self._init_remote_value()
            self._output_remote_value_ref = weakref.ref(ret)
            return ret
        else:
            return self._output_remote_value_ref()

class PerWorkerVariableClosure(ResourceClosure):

    def _init_remote_value(self):
        if False:
            for i in range(10):
                print('nop')
        return RemoteVariable(self, self._output_type_spec)

class _CoordinatedClosureQueue(object):
    """Manage a queue of closures, inflight count and errors from execution.

  This class is thread-safe.
  """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.inflight_closure_count = 0
        self._queue_lock = threading.Lock()
        self._stop_waiting_condition = threading.Condition(self._queue_lock)
        self._closures_queued_condition = threading.Condition(self._queue_lock)
        self._should_process_closures = True
        self._queue_free_slot_condition = threading.Condition(self._queue_lock)
        self._no_inflight_closure_condition = threading.Condition(self._queue_lock)
        self._cancellation_mgr = cancellation.CancellationManager()
        if _CLOSURE_QUEUE_MAX_SIZE <= 0:
            logging.warning('In a `ClusterCoordinator`, creating an infinite closure queue can consume a significant amount of memory and even lead to OOM.')
        self._queue = queue.Queue(maxsize=_CLOSURE_QUEUE_MAX_SIZE)
        metric_utils.monitor_int('queued_closures', self._queue.qsize())
        self._tagged_queue = collections.defaultdict(queue.Queue)
        self._error = None
        self._put_wait_lock = threading.Lock()
        self._watchdog = watchdog.WatchDog(on_triggered=self._on_watchdog_timeout)

    def _on_watchdog_timeout(self):
        if False:
            print('Hello World!')
        logging.info('inflight_closure_count is %d', self._inflight_closure_count)
        logging.info('current error is %s:%r', self._error, self._error)

    @property
    def inflight_closure_count(self):
        if False:
            for i in range(10):
                print('nop')
        return self._inflight_closure_count

    @inflight_closure_count.setter
    def inflight_closure_count(self, value):
        if False:
            while True:
                i = 10
        self._inflight_closure_count = value
        metric_utils.monitor_int('inflight_closures', self._inflight_closure_count)

    def stop(self):
        if False:
            while True:
                i = 10
        with self._queue_lock:
            self._should_process_closures = False
            self._cancellation_mgr.start_cancel()
            self._closures_queued_condition.notify_all()
        self._watchdog.stop()

    def _cancel_all_closures(self):
        if False:
            print('Hello World!')
        'Clears the queue and sets remaining closures cancelled error.\n\n    This method expects self._queue_lock to be held prior to entry.\n    '
        self._cancellation_mgr.start_cancel()
        logging.info('Canceling all closures: waiting for inflight closures to finish')
        while self._inflight_closure_count > 0:
            self._no_inflight_closure_condition.wait()
        logging.info('Canceling all closures: canceling remaining closures on the queue')
        while True:
            try:
                closure = self._queue.get(block=False)
                metric_utils.monitor_int('queued_closures', self._queue.qsize())
                self._queue_free_slot_condition.notify()
                closure.mark_cancelled()
            except queue.Empty:
                break
        self._cancellation_mgr = cancellation.CancellationManager()

    def _raise_if_error(self):
        if False:
            print('Hello World!')
        'Raises the error if one exists.\n\n    If an error exists, cancel the closures in queue, raises it, and clear\n    the error.\n\n    This method expects self._queue_lock to be held prior to entry.\n    '
        if self._error:
            logging.error('Start cancelling closures due to error %r: %s', self._error, self._error)
            self._cancel_all_closures()
            try:
                raise self._error
            finally:
                self._error = None

    def put(self, closure, tag=None):
        if False:
            while True:
                i = 10
        'Put a closure into the queue for later execution.\n\n    If `mark_failed` was called before `put`, the error from the first\n    invocation of `mark_failed` will be raised.\n\n    Args:\n      closure: The `Closure` to put into the queue.\n      tag: if not None, put into a queue with the given tag.\n    '
        closure.tag = tag
        if tag is not None:
            with self._queue_lock:
                self._tagged_queue[tag].put(closure, block=False)
                self._closures_queued_condition.notify_all()
        else:
            with self._put_wait_lock, self._queue_lock:
                self._queue_free_slot_condition.wait_for(lambda : not self._queue.full())
                self._queue.put(closure, block=False)
                metric_utils.monitor_int('queued_closures', self._queue.qsize())
                self._raise_if_error()
                self._closures_queued_condition.notify()

    def get(self, timeout=None, tag=None):
        if False:
            for i in range(10):
                print('nop')
        'Return a closure from the queue to be executed.\n\n    It will try to fetch an item from the queue with the given tag. If this\n    queue is empty, it will then check the global queue.\n\n    Args:\n      timeout: timeout when waiting for a closure to be put.\n      tag: optional tag to specify which queue to query first before querying\n        the global queue.\n\n    Returns:\n      a closure or None after timeout.\n    '
        with self._queue_lock:
            while self._should_process_closures and self._queue.empty() and (tag is None or self._tagged_queue[tag].empty()):
                if not self._closures_queued_condition.wait(timeout=timeout):
                    return None
            if not self._should_process_closures:
                return None
            if tag is not None and (not self._tagged_queue[tag].empty()):
                closure = self._tagged_queue[tag].get(block=False)
                return closure
            closure = self._queue.get(block=False)
            metric_utils.monitor_int('queued_closures', self._queue.qsize())
            assert closure.tag is None
            assert tag is None or self._tagged_queue[tag].empty()
            self._queue_free_slot_condition.notify()
            self.inflight_closure_count += 1
            return closure

    def mark_finished(self):
        if False:
            i = 10
            return i + 15
        'Let the queue know that a closure has been successfully executed.'
        with self._queue_lock:
            if self._inflight_closure_count < 1:
                raise AssertionError('There is no inflight closures to mark_finished.')
            self.inflight_closure_count -= 1
            if self._inflight_closure_count == 0:
                self._no_inflight_closure_condition.notify_all()
            if self._queue.empty() and self._inflight_closure_count == 0:
                self._stop_waiting_condition.notify_all()
            self._watchdog.report_closure_done()

    def put_back(self, closure):
        if False:
            i = 10
            return i + 15
        'Put the closure back into the queue as it was not properly executed.'
        assert closure.tag is None
        with self._queue_lock:
            if self._inflight_closure_count < 1:
                raise AssertionError('There is no inflight closures to put_back.')
            if self._error:
                closure.mark_cancelled()
            else:
                self._queue_free_slot_condition.wait_for(lambda : not self._queue.full())
                self._queue.put(closure, block=False)
                metric_utils.monitor_int('queued_closures', self._queue.qsize())
                self._closures_queued_condition.notify()
            self.inflight_closure_count -= 1
            if self._inflight_closure_count == 0:
                self._no_inflight_closure_condition.notify_all()

    def wait(self, timeout=None):
        if False:
            i = 10
            return i + 15
        'Wait for all closures to be finished before returning.\n\n    If `mark_failed` was called before or during `wait`, the error from the\n    first invocation of `mark_failed` will be raised.\n\n    Args:\n      timeout: A float specifying a timeout for the wait in seconds.\n\n    Returns:\n      True unless the given timeout expired, in which case it returns False.\n    '
        with self._put_wait_lock, self._queue_lock:
            logging.info('Waiting for all global closures to be finished.')
            while not self._error and (not self._queue.empty() or self._inflight_closure_count > 0):
                if not self._stop_waiting_condition.wait(timeout=timeout):
                    return False
            self._raise_if_error()
            return True

    def mark_failed(self, e):
        if False:
            for i in range(10):
                print('nop')
        'Sets error and unblocks any wait() call.'
        with self._queue_lock:
            if self._inflight_closure_count < 1:
                raise AssertionError('There is no inflight closures to mark_failed.')
            if self._error is None:
                self._error = e
            self.inflight_closure_count -= 1
            if self._inflight_closure_count == 0:
                self._no_inflight_closure_condition.notify_all()
            self._stop_waiting_condition.notify_all()

    def done(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns true if the queue is empty and there is no inflight closure.\n\n    If `mark_failed` was called before `done`, the error from the first\n    invocation of `mark_failed` will be raised.\n    '
        with self._queue_lock:
            self._raise_if_error()
            return self._queue.empty() and self._inflight_closure_count == 0

    def clear_tag_unlocked(self, tag):
        if False:
            for i in range(10):
                print('nop')
        self._tagged_queue[tag] = queue.Queue()

class CoordinationServicePreemptionHandler(object):
    """Handles preemptions of workers and parameter servers.

  Starts a thread to regularly poll the coordination service (hosted on PS 0)
  for task states. When a worker's task state reflects an error, it inspects the
  error. If the error is recoverable (i.e. a preemption), it waits for the
  worker to recover, then updates the server def. Otherwise, it raises the error
  to the user.

  A worker error is detected to be recoverable if it is the result of missing a
  heartbeat that workers regularly send to the coordination service.

  The thread also checks for parameter server errors. If these are detected, the
  thread and coordinator shutdown. To resume training in this case, the whole
  job must be restarted and resumed from the latest checkpoint.
  """

    def __init__(self, server_def, cluster):
        if False:
            return 10
        self._server_def = server_def
        self._cluster = cluster
        self._cluster_update_lock = threading.Lock()
        self._cluster_due_for_update_or_finish = threading.Event()
        self._worker_up_cond = threading.Condition(self._cluster_update_lock)
        self._next_task_state_cond = threading.Condition()
        self._task_states = None
        self._error_from_recovery = None
        self._should_preemption_thread_run = True
        self._task_state_poller_thread = utils.RepeatedTimer(interval=_POLL_FREQ_IN_SEC, function=self._get_task_states)
        self._preemption_handler_thread = threading.Thread(target=self._preemption_handler, name='WorkerPreemptionHandler', daemon=True)
        self._preemption_handler_thread.start()
        self._num_workers = self._cluster._num_workers
        self._num_ps = self._cluster._num_ps

    def stop(self):
        if False:
            print('Hello World!')
        'Ensure the worker preemption thread is closed.'
        self._task_state_poller_thread.stop()
        self._should_preemption_thread_run = False
        with self._cluster_update_lock:
            self._cluster_due_for_update_or_finish.set()

    @contextlib.contextmanager
    def wait_on_failure(self, on_failure_fn=None, on_transient_failure_fn=None, on_recovery_fn=None, worker_device_name='(unknown)'):
        if False:
            while True:
                i = 10
        'Catches errors during closure execution and handles them.\n\n    Args:\n      on_failure_fn: an optional function to run if preemption happens.\n      on_transient_failure_fn: an optional function to run if transient failure\n        happens.\n      on_recovery_fn: an optional function to run when a worker is recovered\n        from preemption.\n      worker_device_name: the device name of the worker instance that is passing\n        through the failure.\n\n    Yields:\n      None.\n    '
        assert self._should_preemption_thread_run
        try:
            yield
        except (errors.OpError, ClosureInputError, ClosureAbortedError) as e:
            with self._next_task_state_cond:
                self._next_task_state_cond.wait(_POLL_FREQ_IN_SEC * 1.25)
            with self._next_task_state_cond:
                self._next_task_state_cond.wait(_POLL_FREQ_IN_SEC * 1.25)
            if not self._task_states:
                self._log_ps_failure_and_raise(e, 0)
            worker_states = self._task_states[:self._num_workers]
            ps_states = self._task_states[self._num_workers:]
            if any(ps_states):
                failed_ps_index = [ix for (ix, ps_state) in enumerate(ps_states) if ps_state]
                self._log_ps_failure_and_raise(e, failed_ps_index[0])
            worker_ix = int(worker_device_name.split(':')[-1])
            if worker_states[worker_ix]:
                if self._cluster.closure_queue._cancellation_mgr.is_cancelled:
                    if isinstance(e, errors.CancelledError):
                        raise e
                    else:
                        raise errors.CancelledError(None, None, 'The corresponding function was cancelled while attempting to recover from worker failure.')
                self._handle_failure_and_recovery(e, on_failure_fn, on_transient_failure_fn, on_recovery_fn, worker_device_name)
                return
            if self._cluster._record_and_ignore_transient_timeouts(e):
                logging.error('Remote function on worker %s failed with %r:%s\nThis derived error is ignored and not reported to users.', worker_device_name, e, e)
                if on_transient_failure_fn:
                    on_transient_failure_fn()
                return
            raise e

    def _handle_failure_and_recovery(self, e, on_failure_fn, on_transient_failure_fn, on_recovery_fn, worker_device_name):
        if False:
            return 10
        'Call failure fn, wait for cluster to recover, then call recovery fn.\n\n    Args:\n      e: the Exception thrown during closure execution.\n      on_failure_fn: an optional function to run if preemption happens.\n      on_transient_failure_fn: an optional function to run if transient failure\n        happens.\n      on_recovery_fn: an optional function to run when a worker is recovered\n        from preemption.\n      worker_device_name: the device name of the worker instance that is passing\n        through the failure.\n    '
        if on_failure_fn:
            on_failure_fn(e)
        with self._cluster_update_lock:
            self._cluster_due_for_update_or_finish.set()
            self._worker_up_cond.wait(_WORKER_MAXIMUM_RECOVERY_SEC)
            if self._error_from_recovery:
                try:
                    raise self._error_from_recovery
                finally:
                    self._error_from_recovery = None
            logging.info('Worker %s has been recovered.', worker_device_name)
        if on_recovery_fn:
            logging.info('Worker %s calling on_recovery_fn', worker_device_name)
            with self.wait_on_failure(on_recovery_fn=on_recovery_fn, on_transient_failure_fn=on_transient_failure_fn, worker_device_name=worker_device_name):
                on_recovery_fn()

    def _log_ps_failure_and_raise(self, e, ps_index):
        if False:
            i = 10
            return i + 15
        logging.info('Parameter server failure detected at PS task %d', ps_index)
        self.stop()
        raise PSUnavailableError(e)

    def _get_task_states(self):
        if False:
            print('Hello World!')
        'Get task states and reset to None if coordination service is down.'
        try:
            self._task_states = context.context().get_task_states([('worker', self._num_workers), ('ps', self._num_ps)])
        except (errors.UnavailableError, errors.InternalError) as e:
            if isinstance(e, errors.InternalError) and 'coordination service is not enabled' not in str(e).lower():
                raise
            self._task_states = None
        with self._next_task_state_cond:
            self._next_task_state_cond.notify_all()

    def _preemption_handler(self):
        if False:
            return 10
        'A loop that handles preemption.\n\n    This loop waits for signal of worker preemption and upon worker preemption,\n    it waits until all workers are back and updates the cluster about the\n    restarted workers.\n    '
        assert self._should_preemption_thread_run
        while True:
            self._cluster_due_for_update_or_finish.wait()
            if not self._should_preemption_thread_run:
                logging.info('Stopping the failure handing thread.')
                break
            with self._cluster_update_lock:
                try:
                    logging.info('Cluster now being recovered.')
                    context.context().update_server_def(self._server_def)
                    logging.info('Cluster successfully recovered.')
                    self._notify_cluster_update()
                except Exception as e:
                    logging.info('Error occurred while updating server def: %s', e)
                    with self._next_task_state_cond:
                        self._next_task_state_cond.wait(_POLL_FREQ_IN_SEC * 2)
                    if not self._task_states:
                        self._error_from_recovery = e
                    else:
                        ps_states = self._task_states[self._num_workers:]
                        if any(ps_states):
                            self._error_from_recovery = e
                    self._notify_cluster_update()
                    logging.error('Cluster update failed with error: %s. Retrying...', e)

    def _notify_cluster_update(self):
        if False:
            print('Hello World!')
        self._worker_up_cond.notify_all()
        if self._should_preemption_thread_run:
            self._cluster_due_for_update_or_finish.clear()

class WorkerPreemptionHandler(object):
    """Handles worker preemptions."""

    def __init__(self, server_def, cluster):
        if False:
            while True:
                i = 10
        self._server_def = server_def
        self._cluster = cluster
        self._cluster_update_lock = threading.Lock()
        self._cluster_due_for_update_or_finish = threading.Event()
        self._worker_up_cond = threading.Condition(self._cluster_update_lock)
        self._error_from_recovery = None
        self._should_preemption_thread_run = True
        self._preemption_handler_thread = threading.Thread(target=self._preemption_handler, name='WorkerPreemptionHandler', daemon=True)
        self._preemption_handler_thread.start()

    def stop(self):
        if False:
            return 10
        'Ensure the worker preemption thread is closed.'
        self._should_preemption_thread_run = False
        with self._cluster_update_lock:
            self._cluster_due_for_update_or_finish.set()

    def _validate_preemption_failure(self, e):
        if False:
            while True:
                i = 10
        'Validates that the given exception represents worker preemption.'
        if _is_worker_failure(e) and (not self._cluster.closure_queue._cancellation_mgr.is_cancelled):
            metric_utils.monitor_increment_counter('worker_failures')
            return
        raise e

    @contextlib.contextmanager
    def wait_on_failure(self, on_failure_fn=None, on_transient_failure_fn=None, on_recovery_fn=None, worker_device_name='(unknown)'):
        if False:
            return 10
        'Catches worker preemption error and wait until failed workers are back.\n\n    Args:\n      on_failure_fn: an optional function to run if preemption happens.\n      on_transient_failure_fn: an optional function to run if transient failure\n        happens.\n      on_recovery_fn: an optional function to run when a worker is recovered\n        from preemption.\n      worker_device_name: the device name of the worker instance that is passing\n        through the failure.\n\n    Yields:\n      None.\n    '
        assert self._should_preemption_thread_run
        try:
            yield
        except (errors.OpError, ClosureInputError, ClosureAbortedError, TypeError) as e:
            if self._cluster._record_and_ignore_transient_ps_failure(e):
                logging.error('Remote function on worker %s failed with %r:%s\nIt is treated as a transient connectivity failure for now.', worker_device_name, e, e)
                if on_transient_failure_fn:
                    on_transient_failure_fn()
                return
            if self._cluster._record_and_ignore_transient_timeouts(e):
                logging.error('Remote function on worker %s failed with %r:%s\nThis derived error is ignored and not reported to users.', worker_device_name, e, e)
                if on_transient_failure_fn:
                    on_transient_failure_fn()
                return
            if isinstance(e, errors.CancelledError) and '/job:' in str(e):
                logging.error('Remote function on worker %s failed with %r:%s\nThis derived error is ignored and not reported to users.', worker_device_name, e, e)
                if on_transient_failure_fn:
                    on_transient_failure_fn()
                return
            self._validate_preemption_failure(e)
            logging.error('Worker %s failed with %r:%s', worker_device_name, e, e)
            if on_failure_fn:
                on_failure_fn(e)
            with self._cluster_update_lock:
                self._cluster_due_for_update_or_finish.set()
                self._worker_up_cond.wait(_WORKER_MAXIMUM_RECOVERY_SEC)
                if self._error_from_recovery:
                    try:
                        raise self._error_from_recovery
                    finally:
                        self._error_from_recovery = None
                logging.info('Worker %s has been recovered.', worker_device_name)
            if on_recovery_fn:
                logging.info('Worker %s calling on_recovery_fn', worker_device_name)
                with self.wait_on_failure(on_recovery_fn=on_recovery_fn, on_transient_failure_fn=on_transient_failure_fn, worker_device_name=worker_device_name):
                    on_recovery_fn()

    def _preemption_handler(self):
        if False:
            i = 10
            return i + 15
        'A loop that handles preemption.\n\n    This loop waits for signal of worker preemption and upon worker preemption,\n    it waits until all workers are back and updates the cluster about the\n    restarted workers.\n    '
        assert self._should_preemption_thread_run
        while True:
            self._cluster_due_for_update_or_finish.wait()
            if not self._should_preemption_thread_run:
                logging.info('Stopping the failure handing thread.')
                break
            with self._cluster_update_lock:
                try:
                    logging.info('Cluster now being recovered.')
                    with metric_utils.monitored_timer('server_def_update'):
                        context.context().update_server_def(self._server_def)
                    logging.info('Cluster successfully recovered.')
                    self._worker_up_cond.notify_all()
                    if self._should_preemption_thread_run:
                        self._cluster_due_for_update_or_finish.clear()
                except Exception as e:
                    logging.info('Error occurred while updating server def: %s', e)
                    try:
                        self._validate_preemption_failure(e)
                    except Exception as ps_e:
                        logging.info('Error that occurred while updating server def is not a worker failure. So set it as _error_from_recovery')
                        self._error_from_recovery = ps_e
                        self._worker_up_cond.notify_all()
                        if self._should_preemption_thread_run:
                            self._cluster_due_for_update_or_finish.clear()
                    logging.error('Cluster update failed with error: %s. Retrying...', e)

class Worker(object):
    """A worker in a cluster.

  Attributes:
    worker_index: The index of the worker in the cluster.
    device_name: The device string of the worker, e.g. "/job:worker/task:1".
    executor: The worker's executor for remote function execution.
    failure_handler: The failure handler used to handler worker preemption
      failure.
  """

    def __init__(self, worker_index, device_name, cluster):
        if False:
            for i in range(10):
                print('nop')
        self.worker_index = worker_index
        self.device_name = device_name
        self.executor = executor.new_executor(enable_async=False)
        self.failure_handler = cluster.failure_handler
        self._cluster = cluster
        self._resource_tracking_lock = threading.Lock()
        self._resource_remote_value_refs = []
        self._is_dead_with_error = None
        self._should_worker_thread_run = True
        threading.Thread(target=self._process_queue, name='WorkerClosureProcessingLoop-%d' % self.worker_index, daemon=True).start()

    def stop(self):
        if False:
            print('Hello World!')
        'Ensure the worker thread is closed.'
        self._should_worker_thread_run = False

    def _schedule_resource(self, closure):
        if False:
            while True:
                i = 10
        self._cluster.closure_queue.put(closure, tag=self.worker_index)

    def _set_resources_aborted(self, e):
        if False:
            return 10
        'Set the resource ABORTED and add an error to it.'
        logging.info('[Worker %d] Clearing all resources.', self.worker_index)
        for weakref_resource in self._resource_remote_value_refs:
            resource = weakref_resource()
            if resource:
                resource._set_aborted(ClosureAbortedError(e))

    def _on_closure_failure(self, closure, e):
        if False:
            return 10
        logging.info('[Worker %d] Putting back a closure after it failed.', self.worker_index)
        self._cluster.closure_queue.put_back(closure)
        with self._resource_tracking_lock:
            self._is_dead_with_error = e
            self._set_resources_aborted(e)

    def _on_resource_closure_failure(self, e):
        if False:
            i = 10
            return i + 15
        'Clear tagged queue to ensure resource closures are rebuilt.\n\n    Args:\n      e: The exception arisen from the resource closure.\n    '
        logging.info('[Worker %d] Clearing tagged queue after resource closure failure.', self.worker_index)
        with self._resource_tracking_lock:
            self._is_dead_with_error = e
            self._cluster.closure_queue.clear_tag_unlocked(self.worker_index)
            self._set_resources_aborted(e)

    def _on_worker_recovery(self):
        if False:
            while True:
                i = 10
        logging.info('[Worker %d] calling _on_worker_recovery', self.worker_index)
        with self._resource_tracking_lock:
            for weakref_resource in self._resource_remote_value_refs:
                resource = weakref_resource()
                if resource:
                    self._schedule_resource(resource._closure)
            self._is_dead_with_error = False

    def _process_closure(self, closure):
        if False:
            for i in range(10):
                print('nop')
        'Runs a closure with preemption handling.'
        try:
            with self.failure_handler.wait_on_failure(on_failure_fn=lambda e: self._on_closure_failure(closure, e), on_transient_failure_fn=lambda : self._cluster.closure_queue.put_back(closure), on_recovery_fn=self._on_worker_recovery, worker_device_name=self.device_name):
                closure.execute_on(self)
                with metric_utils.monitored_timer('remote_value_fetch'):
                    closure.maybe_call_with_output_remote_value(lambda r: r.get())
                self._cluster.closure_queue.mark_finished()
        except Exception as e:
            if not isinstance(e, errors.CancelledError):
                logging.error(' /job:worker/task:%d encountered the following error when processing closure: %r:%s', self.worker_index, e, e)
            closure.maybe_call_with_output_remote_value(lambda r: r._set_error(e))
            self._cluster.closure_queue.mark_failed(e)

    def _process_resource_closure(self, closure):
        if False:
            i = 10
            return i + 15
        'Run the given resource closure with preemption handling.'
        assert closure.tag == self.worker_index
        try:
            with self.failure_handler.wait_on_failure(on_failure_fn=self._on_resource_closure_failure, on_transient_failure_fn=lambda : self._process_resource_closure(closure), on_recovery_fn=self._on_worker_recovery, worker_device_name=self.device_name):
                closure.execute_on(self)
        except Exception as e:
            logging.info('[Worker %d] got an exception when processing resource closure', self.worker_index)
            if not isinstance(e, errors.CancelledError):
                logging.error(' /job:worker/task:%d encountered the following error when processing resource closure: %r:%s', self.worker_index, e, e)
            closure.maybe_call_with_output_remote_value(lambda r: r._set_error(e))

    def _maybe_delay(self):
        if False:
            print('Hello World!')
        'Delay if corresponding env vars are set.'
        delay_secs = int(os.environ.get('TF_COORDINATOR_SCHEDULE_START_DELAY', '0'))
        delay_secs *= self.worker_index
        delay_cap = int(os.environ.get('TF_COORDINATOR_SCHEDULE_START_DELAY_MAX', '0'))
        if delay_cap:
            delay_secs = min(delay_secs, delay_cap)
        if delay_secs > 0:
            logging.info(' Worker %d sleeping for %d seconds before running function', self.worker_index, delay_secs)
        time.sleep(delay_secs)

    def _process_queue(self):
        if False:
            while True:
                i = 10
        'Function running in a worker thread to process closure queues.'
        self._maybe_delay()
        while self._should_worker_thread_run:
            closure = self._cluster.closure_queue.get(tag=self.worker_index)
            if not self._should_worker_thread_run or closure is None:
                if closure is not None:
                    closure.mark_cancelled()
                return
            if isinstance(closure, ResourceClosure):
                self._process_resource_closure(closure)
            else:
                self._process_closure(closure)
            del closure

    def create_resource(self, function, args=None, kwargs=None):
        if False:
            while True:
                i = 10
        'Asynchronously creates a per-worker resource represented by a `RemoteValue`.\n\n    Args:\n      function: the resource function to be run remotely. It should be a\n        `tf.function`, a concrete function or a Python function.\n      args: positional arguments to be passed to the function.\n      kwargs: keyword arguments to be passed to the function.\n\n    Returns:\n      one or several RemoteValue objects depending on the function return\n      values.\n    '
        closure = ResourceClosure(function, self._cluster.resource_cancellation_mgr, args=args, kwargs=kwargs)
        return self._register_and_schedule_resource_closure(closure)

    def create_variable_resource(self, function, args=None, kwargs=None):
        if False:
            for i in range(10):
                print('nop')
        'Create a per-worker variable.'
        closure = PerWorkerVariableClosure(function, self._cluster.resource_cancellation_mgr, args=args, kwargs=kwargs)
        return self._register_and_schedule_resource_closure(closure)

    def _register_and_schedule_resource_closure(self, closure):
        if False:
            for i in range(10):
                print('nop')
        'Build remote value for, register for reconstruction, and schedule.'
        resource_remote_value = closure.build_output_remote_value()
        with self._resource_tracking_lock:
            self._register_resource(resource_remote_value)
            if self._is_dead_with_error:
                resource_remote_value._set_aborted(ClosureAbortedError(self._is_dead_with_error))
            else:
                self._schedule_resource(closure)
        return resource_remote_value

    def _register_resource(self, resource_remote_value):
        if False:
            i = 10
            return i + 15
        if not isinstance(resource_remote_value, RemoteValue):
            raise ValueError('Resource being registered is not of type `tf.distribute.experimental.coordinator.RemoteValue`.')
        self._resource_remote_value_refs.append(weakref.ref(resource_remote_value))

class Cluster(object):
    """A cluster with workers.

  We assume all function errors are fatal and based on this assumption our
  error reporting logic is:
  1) Both `schedule` and `join` can raise a non-retryable error which is the
  first error seen by the coordinator from any previously scheduled functions.
  2) When an error is raised, there is no guarantee on how many previously
  scheduled functions have been executed; functions that have not been executed
  will be thrown away and marked as cancelled.
  3) After an error is raised, the internal state of error will be cleared.
  I.e. functions can continue to be scheduled and subsequent calls of `schedule`
  or `join` will not raise the same error again.

  Attributes:
    failure_handler: The failure handler used to handler worker preemption
      failure.
    workers: a list of `Worker` objects in the cluster.
    closure_queue: the global Closure queue.
    resource_cancellation_mgr: the cancellation manager used to cancel resource
      closures.
  """

    def __init__(self, strategy):
        if False:
            print('Hello World!')
        'Initializes the cluster instance.'
        self._num_workers = strategy._num_workers
        self._num_ps = strategy._num_ps
        self._transient_ps_failures_threshold = int(os.environ.get('TF_COORDINATOR_IGNORE_TRANSIENT_PS_FAILURES', 3))
        self._potential_ps_failures_lock = threading.Lock()
        self._potential_ps_failures_count = [0] * self._num_ps
        self._transient_timeouts_threshold = int(os.environ.get('TF_COORDINATOR_IGNORE_TRANSIENT_TIMEOUTS', self._num_workers // 10))
        self._transient_timeouts_lock = threading.Lock()
        self._transient_timeouts_count = 0
        self.closure_queue = _CoordinatedClosureQueue()
        if os.getenv('TF_PSS_ENABLE_COORDINATION_SERVICE'):
            self.failure_handler = CoordinationServicePreemptionHandler(context.get_server_def(), self)
        else:
            self.failure_handler = WorkerPreemptionHandler(context.get_server_def(), self)
        worker_device_strings = ['/job:worker/replica:0/task:%d' % i for i in range(self._num_workers)]
        self.workers = [Worker(i, w, self) for (i, w) in enumerate(worker_device_strings)]
        self.resource_cancellation_mgr = cancellation.CancellationManager()

    def stop(self):
        if False:
            while True:
                i = 10
        'Stop worker, worker preemption threads, and the closure queue.'
        logging.info('Stopping cluster, starting with failure handler')
        self.failure_handler.stop()
        logging.info('Stopping workers')
        for worker in self.workers:
            worker.stop()
        logging.info('Stopping queue')
        self.closure_queue.stop()
        logging.info('Start cancelling remote resource-building functions')
        self.resource_cancellation_mgr.start_cancel()

    def _record_and_ignore_transient_ps_failure(self, e):
        if False:
            print('Hello World!')
        'Records potential PS failures and return if failure should be ignored.'
        if self._transient_ps_failures_threshold <= 0 or not _is_ps_failure(e):
            return False
        ps_tasks = _extract_failed_ps_instances(str(e))
        with self._potential_ps_failures_lock:
            for t in ps_tasks:
                self._potential_ps_failures_count[t] += 1
                if self._potential_ps_failures_count[t] >= self._transient_ps_failures_threshold:
                    return False
        return True

    def _record_and_ignore_transient_timeouts(self, e):
        if False:
            print('Hello World!')
        'Records observed timeout error and return if it should be ignored.'
        if self._transient_timeouts_threshold <= 0:
            return False
        if not isinstance(e, errors.DeadlineExceededError):
            return False
        with self._transient_timeouts_lock:
            self._transient_timeouts_count += 1
            if self._transient_timeouts_count >= self._transient_timeouts_threshold:
                return False
        return True

    def schedule(self, function, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Schedules `function` to be dispatched to a worker for execution.\n\n    Args:\n      function: The function to be dispatched to a worker for execution\n        asynchronously.\n      args: Positional arguments for `fn`.\n      kwargs: Keyword arguments for `fn`.\n\n    Returns:\n      A `RemoteValue` object.\n    '
        closure = Closure(function, self.closure_queue._cancellation_mgr, args=args, kwargs=kwargs)
        ret = closure.build_output_remote_value()
        self.closure_queue.put(closure)
        return ret

    def join(self):
        if False:
            i = 10
            return i + 15
        'Blocks until all scheduled functions are executed.'
        self.closure_queue.wait()

    def done(self):
        if False:
            return 10
        'Returns true if all scheduled functions are executed.'
        return self.closure_queue.done()

@tf_export('distribute.experimental.coordinator.ClusterCoordinator', 'distribute.coordinator.ClusterCoordinator', v1=[])
class ClusterCoordinator(object):
    """An object to schedule and coordinate remote function execution.

  This class is used to create fault-tolerant resources and dispatch functions
  to remote TensorFlow servers.

  Currently, this class is not supported to be used in a standalone manner. It
  should be used in conjunction with a `tf.distribute` strategy that is designed
  to work with it. The `ClusterCoordinator` class currently only works
  `tf.distribute.experimental.ParameterServerStrategy`.

  __The `schedule`/`join` APIs__

  The most important APIs provided by this class is the `schedule`/`join` pair.
  The `schedule` API is non-blocking in that it queues a `tf.function` and
  returns a `RemoteValue` immediately. The queued functions will be dispatched
  to remote workers in background threads and their `RemoteValue`s will be
  filled asynchronously. Since `schedule` doesn’t require worker assignment, the
  `tf.function` passed in can be executed on any available worker. If the worker
  it is executed on becomes unavailable before its completion, it will be
  migrated to another worker. Because of this fact and function execution is not
  atomic, a function may be executed more than once.

  __Handling Task Failure__

  This class when used with
  `tf.distribute.experimental.ParameterServerStrategy`, comes with built-in
  fault tolerance for worker failures. That is, when some workers are not
  available for any reason to be reached from the coordinator, the training
  progress continues to be made with the remaining workers. Upon recovery of a
  failed worker, it will be added for function execution after datasets created
  by `create_per_worker_dataset` are re-built on it.

  When a parameter server fails, a `tf.errors.UnavailableError` is raised by
  `schedule`, `join` or `done`. In this case, in addition to bringing back the
  failed parameter server, users should restart the coordinator so that it
  reconnects to workers and parameter servers, re-creates the variables, and
  loads checkpoints. If the coordinator fails, after the user brings it back,
  the program will automatically connect to workers and parameter servers, and
  continue the progress from a checkpoint.

  It is thus essential that in user's program, a checkpoint file is periodically
  saved, and restored at the start of the program. If an
  `tf.keras.optimizers.Optimizer` is checkpointed, after restoring from a
  checkpoiont, its `iterations` property roughly indicates the number of steps
  that have been made. This can be used to decide how many epochs and steps are
  needed before the training completion.

  See `tf.distribute.experimental.ParameterServerStrategy` docstring for an
  example usage of this API.

  This is currently under development, and the API as well as implementation
  are subject to changes.
  """

    def __new__(cls, strategy):
        if False:
            i = 10
            return i + 15
        if strategy._cluster_coordinator is None:
            strategy._cluster_coordinator = super(ClusterCoordinator, cls).__new__(cls)
        return strategy._cluster_coordinator

    def __init__(self, strategy):
        if False:
            while True:
                i = 10
        'Initialization of a `ClusterCoordinator` instance.\n\n    Args:\n      strategy: a supported `tf.distribute.Strategy` object. Currently, only\n        `tf.distribute.experimental.ParameterServerStrategy` is supported.\n\n    Raises:\n      ValueError: if the strategy being used is not supported.\n    '
        if not getattr(self, '_has_initialized', False):
            if not hasattr(strategy, '_is_parameter_server_strategy_v2'):
                raise ValueError('Only `tf.distribute.experimental.ParameterServerStrategy` is supported to work with `tf.distribute.experimental.coordinator.ClusterCoordinator` currently.')
            self._strategy = strategy
            self.strategy.extended._used_with_coordinator = True
            self._cluster = Cluster(strategy)
            self._has_initialized = True

    def __del__(self):
        if False:
            return 10
        logging.info('ClusterCoordinator destructor: stopping cluster')
        self._cluster.stop()

    @property
    def strategy(self):
        if False:
            while True:
                i = 10
        'Returns the `Strategy` associated with the `ClusterCoordinator`.'
        return self._strategy

    def schedule(self, fn, args=None, kwargs=None):
        if False:
            return 10
        'Schedules `fn` to be dispatched to a worker for asynchronous execution.\n\n    This method is non-blocking in that it queues the `fn` which will be\n    executed later and returns a\n    `tf.distribute.experimental.coordinator.RemoteValue` object immediately.\n    `fetch` can be called on it to wait for the function execution to finish\n    and retrieve its output from a remote worker. On the other hand, call\n    `tf.distribute.experimental.coordinator.ClusterCoordinator.join` to wait for\n    all scheduled functions to finish.\n\n    `schedule` guarantees that `fn` will be executed on a worker at least once;\n    it could be more than once if its corresponding worker fails in the middle\n    of its execution. Note that since worker can fail at any point when\n    executing the function, it is possible that the function is partially\n    executed, but `tf.distribute.experimental.coordinator.ClusterCoordinator`\n    guarantees that in those events, the function will eventually be executed on\n    any worker that is available.\n\n    If any previously scheduled function raises an error, `schedule` will raise\n    any one of those errors, and clear the errors collected so far. What happens\n    here, some of the previously scheduled functions may have not been executed.\n    User can call `fetch` on the returned\n    `tf.distribute.experimental.coordinator.RemoteValue` to inspect if they have\n    executed, failed, or cancelled, and reschedule the corresponding function if\n    needed.\n\n    When `schedule` raises, it guarantees that there is no function that is\n    still being executed.\n\n    At this time, there is no support of worker assignment for function\n    execution, or priority of the workers.\n\n    `args` and `kwargs` are the arguments passed into `fn`, when `fn` is\n    executed on a worker. They can be\n    `tf.distribute.experimental.coordinator.PerWorkerValues` and in this case,\n    the argument will be substituted with the corresponding component on the\n    target worker. Arguments that are not\n    `tf.distribute.experimental.coordinator.PerWorkerValues` will be passed into\n    `fn` as-is. Currently, `tf.distribute.experimental.coordinator.RemoteValue`\n    is not supported to be input `args` or `kwargs`.\n\n    Args:\n      fn: A `tf.function`; the function to be dispatched to a worker for\n        execution asynchronously. Regular python function is not supported to be\n        scheduled.\n      args: Positional arguments for `fn`.\n      kwargs: Keyword arguments for `fn`.\n\n    Returns:\n      A `tf.distribute.experimental.coordinator.RemoteValue` object that\n      represents the output of the function scheduled.\n\n    Raises:\n      Exception: one of the exceptions caught by the coordinator from any\n        previously scheduled function, since the last time an error was thrown\n        or since the beginning of the program.\n    '
        if not isinstance(fn, (def_function.Function, tf_function.ConcreteFunction)):
            raise TypeError('`tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` only accepts a `tf.function` or a concrete function.')
        with self.strategy.scope():
            self.strategy.extended._being_scheduled = True
            schedule_remote_value = self._cluster.schedule(fn, args=args, kwargs=kwargs)
            self.strategy.extended._being_scheduled = False
            return schedule_remote_value

    def join(self):
        if False:
            while True:
                i = 10
        'Blocks until all the scheduled functions have finished execution.\n\n    If any previously scheduled function raises an error, `join` will fail by\n    raising any one of those errors, and clear the errors collected so far. If\n    this happens, some of the previously scheduled functions may have not been\n    executed. Users can call `fetch` on the returned\n    `tf.distribute.experimental.coordinator.RemoteValue` to inspect if they have\n    executed, failed, or cancelled. If some that have been cancelled need to be\n    rescheduled, users should call `schedule` with the function again.\n\n    When `join` returns or raises, it guarantees that there is no function that\n    is still being executed.\n\n    Raises:\n      Exception: one of the exceptions caught by the coordinator by any\n        previously scheduled function since the last time an error was thrown or\n        since the beginning of the program.\n    '
        self._cluster.join()

    def done(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns whether all the scheduled functions have finished execution.\n\n    If any previously scheduled function raises an error, `done` will fail by\n    raising any one of those errors.\n\n    When `done` returns True or raises, it guarantees that there is no function\n    that is still being executed.\n\n    Returns:\n      Whether all the scheduled functions have finished execution.\n    Raises:\n      Exception: one of the exceptions caught by the coordinator by any\n        previously scheduled function since the last time an error was thrown or\n        since the beginning of the program.\n    '
        return self._cluster.done()

    def create_per_worker_dataset(self, dataset_fn):
        if False:
            for i in range(10):
                print('nop')
        'Create dataset on each worker.\n\n    This creates dataset on workers from the input which can be either a\n    `tf.data.Dataset`, a `tf.distribute.DistributedDataset` or a function which\n    returns a dataset, and returns an object that represents the collection of\n    those individual datasets. Calling `iter` on such collection of datasets\n    returns a `tf.distribute.experimental.coordinator.PerWorkerValues`, which is\n    a collection of iterators, where the iterators have been placed on\n    respective workers.\n\n    Calling `next` on a `PerWorkerValues` of iterator is unsupported. The\n    iterator is meant to be passed as an argument into\n    `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule`. When\n    the scheduled function is about to be executed by a worker, the\n    function will receive the individual iterator that corresponds to the\n    worker. The `next` method can be called on an iterator inside a\n    scheduled function when the iterator is an input of the function.\n\n    Currently the `schedule` method assumes workers are all the same and thus\n    assumes the datasets on different workers are the same, except they may be\n    shuffled differently if they contain a `dataset.shuffle` operation and a\n    random seed is not set. Because of this, we also recommend the datasets to\n    be repeated indefinitely and schedule a finite number of steps instead of\n    relying on the `OutOfRangeError` from a dataset.\n\n\n    Example:\n\n    ```python\n    strategy = tf.distribute.experimental.ParameterServerStrategy(\n        cluster_resolver=...)\n    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(\n        strategy=strategy)\n\n    @tf.function\n    def worker_fn(iterator):\n      return next(iterator)\n\n    def per_worker_dataset_fn():\n      return strategy.distribute_datasets_from_function(\n          lambda x: tf.data.Dataset.from_tensor_slices([3] * 3))\n\n    per_worker_dataset = coordinator.create_per_worker_dataset(\n        per_worker_dataset_fn)\n    per_worker_iter = iter(per_worker_dataset)\n    remote_value = coordinator.schedule(worker_fn, args=(per_worker_iter,))\n    assert remote_value.fetch() == 3\n    ```\n\n    Args:\n      dataset_fn: The dataset function that returns a dataset. This is to be\n        executed on the workers.\n\n    Returns:\n      An object that represents the collection of those individual\n      datasets. `iter` is expected to be called on this object that returns\n      a `tf.distribute.experimental.coordinator.PerWorkerValues` of the\n      iterators (that are on the workers).\n    '
        return values_lib.get_per_worker_dataset(dataset_fn, self)

    def _create_per_worker_resources(self, fn, args=None, kwargs=None):
        if False:
            i = 10
            return i + 15
        'Synchronously create resources on the workers.\n\n    The resources are represented by\n    `tf.distribute.experimental.coordinator.RemoteValue`s.\n\n    Args:\n      fn: The function to be dispatched to all workers for execution\n        asynchronously.\n      args: Positional arguments for `fn`.\n      kwargs: Keyword arguments for `fn`.\n\n    Returns:\n      A `tf.distribute.experimental.coordinator.PerWorkerValues` object, which\n      wraps a tuple of `tf.distribute.experimental.coordinator.RemoteValue`\n      objects.\n    '
        results = []
        for w in self._cluster.workers:
            results.append(w.create_resource(fn, args=args, kwargs=kwargs))
        return PerWorkerValues(tuple(results))

    def _create_per_worker_variables(self, fn, args=None, kwargs=None):
        if False:
            return 10
        'Asynchronously create variables on workers.'
        results = []
        for w in self._cluster.workers:
            results.append(w.create_variable_resource(fn, args=args, kwargs=kwargs))
        return PerWorkerValues(tuple(results))

    def fetch(self, val):
        if False:
            return 10
        'Blocking call to fetch results from the remote values.\n\n    This is a wrapper around\n    `tf.distribute.experimental.coordinator.RemoteValue.fetch` for a\n    `RemoteValue` structure; it returns the execution results of\n    `RemoteValue`s. If not ready, wait for them while blocking the caller.\n\n    Example:\n    ```python\n    strategy = ...\n    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(\n        strategy)\n\n    def dataset_fn():\n      return tf.data.Dataset.from_tensor_slices([1, 1, 1])\n\n    with strategy.scope():\n      v = tf.Variable(initial_value=0)\n\n    @tf.function\n    def worker_fn(iterator):\n      def replica_fn(x):\n        v.assign_add(x)\n        return v.read_value()\n      return strategy.run(replica_fn, args=(next(iterator),))\n\n    distributed_dataset = coordinator.create_per_worker_dataset(dataset_fn)\n    distributed_iterator = iter(distributed_dataset)\n    result = coordinator.schedule(worker_fn, args=(distributed_iterator,))\n    assert coordinator.fetch(result) == 1\n    ```\n\n    Args:\n      val: The value to fetch the results from. If this is structure of\n        `tf.distribute.experimental.coordinator.RemoteValue`, `fetch()` will be\n        called on the individual\n        `tf.distribute.experimental.coordinator.RemoteValue` to get the result.\n\n    Returns:\n      If `val` is a `tf.distribute.experimental.coordinator.RemoteValue` or a\n      structure of `tf.distribute.experimental.coordinator.RemoteValue`s,\n      return the fetched `tf.distribute.experimental.coordinator.RemoteValue`\n      values immediately if they are available, or block the call until they are\n      available, and return the fetched\n      `tf.distribute.experimental.coordinator.RemoteValue` values with the same\n      structure. If `val` is other types, return it as-is.\n    '

        def _maybe_fetch(val):
            if False:
                return 10
            if isinstance(val, RemoteValue):
                return val.fetch()
            else:
                return val
        return nest.map_structure(_maybe_fetch, val)

def _extract_failed_ps_instances(err_msg):
    if False:
        print('Hello World!')
    'Return a set of potentially failing ps instances from error message.'
    tasks = re.findall('/job:ps/replica:0/task:[0-9]+', err_msg)
    return set((int(t.split(':')[-1]) for t in tasks))

def _is_ps_failure(error):
    if False:
        return 10
    'Whether the error is considered a parameter server failure.'
    if isinstance(error, PSUnavailableError):
        return True
    if isinstance(error, (ClosureInputError, ClosureAbortedError)):
        error = error.original_exception
    if _RPC_ERROR_FROM_PS not in str(error):
        return False
    if isinstance(error, (errors.UnavailableError, errors.AbortedError)):
        return True
    if isinstance(error, errors.InvalidArgumentError):
        if 'unknown device' in str(error).lower() or 'Unable to find the relevant tensor remote_handle' in str(error):
            return True
    return False

def _handle_graph_execution_error_as_worker_failure():
    if False:
        return 10
    return int(os.environ.get('TF_PS_HANDLE_UNKNOWN_ERROR', '0')) > 0

def _is_worker_failure(error):
    if False:
        for i in range(10):
            print('nop')
    'Whether the error is considered a worker failure.'
    if _handle_graph_execution_error_as_worker_failure() and isinstance(error, errors.UnknownError) and ('Graph execution error' in str(error)):
        logging.info(f'Handling {type(error)}: {str(error)} as worker failure.')
        return True
    if isinstance(error, (ClosureInputError, ClosureAbortedError)):
        error = error.original_exception
    if _JOB_WORKER_STRING_IDENTIFIER not in str(error):
        return False
    if _RPC_ERROR_FROM_PS in str(error):
        return False
    if isinstance(error, (errors.UnavailableError, errors.AbortedError)):
        return True
    if isinstance(error, errors.InvalidArgumentError):
        if 'unknown device' in str(error).lower() or 'Primary device is not remote' in str(error) or 'Unable to find the relevant tensor remote_handle' in str(error):
            return True
    if isinstance(error, errors.NotFoundError):
        if 'is neither a type of a primitive operation nor a name of a function registered' in str(error):
            return True
    if isinstance(error, errors.CancelledError):
        return True
    if isinstance(error, TypeError) and 'Binding inputs to tf.function' in str(error):
        return True
    return False