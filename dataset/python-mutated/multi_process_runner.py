"""Multi-process runner for testing purpose."""
import collections
import contextlib
import json
import os
import signal
import sys
import threading
import time
import unittest
import weakref
from absl import logging
import six
from six.moves import queue as Queue
from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.util.tf_export import tf_export
multiprocessing = multi_process_lib.multiprocessing
try:
    import faulthandler
except ImportError:
    faulthandler = None
try:
    import dill
except ImportError:
    dill = None
try:
    import tblib.pickling_support
    tblib.pickling_support.install()
except ImportError:
    pass
_ProcessStatusInfo = collections.namedtuple('_ProcessStatusInfo', ['task_type', 'task_id', 'is_successful', 'exc_info', 'return_value'])
MultiProcessRunnerResult = collections.namedtuple('MultiProcessRunnerResult', ['return_value', 'stdout'])
TestEnvironment = collections.namedtuple('TestEnvironment', ['task_type', 'task_id', 'cluster_spec', 'rpc_layer', 'grpc_fail_fast', 'v2_enabled', 'executing_eagerly', 'visible_gpus'])
Resources = collections.namedtuple('Resources', ['process_status_queue', 'parent_to_sub_queue', 'streaming_pipe_w', 'barrier'])
_DEFAULT_TIMEOUT_SEC = 200
_FORCE_KILL_WAIT_SEC = 30

class MultiProcessRunner(object):
    """A utility class to start multiple processes to simulate a cluster.

  We need to use multiple processes to simulate a cluster in TF 2.0 tests
  because TF 2.0 has some process-global data structures that have to be
  separated by processes. We also need child processes to test out our fault
  tolerance because shutting down a standard TensorFlow server within its
  process is not supported.

  Note: the main test program that uses this runner class must run main program
  via `test_main` defined in this file. Using this runner in non-test binaries
  is not supported yet.

  This class is not thread-safe. Child processes will inherit TF2 behavior flag.
  """

    def __init__(self, fn, cluster_spec, rpc_layer=None, max_run_time=None, grpc_fail_fast=None, stream_output=True, return_output=False, use_dill_for_args=True, daemon=False, dependence_on_chief=True, auto_restart=False, share_gpu=True, args=None, kwargs=None):
        if False:
            i = 10
            return i + 15
        'Instantiation of a `MultiProcessRunner`.\n\n    Args:\n      fn: Function to be run on child processes. This will be run on processes\n        for all task types.\n      cluster_spec: Dict for cluster spec. The utility function\n        `tf.__internal__.distribute.multi_process_runner.create_cluster_spec`\n        can be conveniently used to create such dict. The following is an\n        example of cluster with three workers and two ps\'s.\n        {"worker": ["worker0.example.com:2222",\n                    "worker1.example.com:2222",\n                    "worker2.example.com:2222"],\n         "ps": ["ps0.example.com:2222",\n                "ps1.example.com:2222"]}\n      rpc_layer: RPC layer to use. Default value is \'grpc\'.\n      max_run_time: `None` or integer. If not `None`, child processes are forced\n        to exit at approximately this many seconds after this utility is called.\n        We achieve this through `signal.alarm()` api. Note that this is best\n        effort at Python level since Python signal handler does not get executed\n        when it runs lower level C/C++ code. So it can be delayed for\n        arbitrarily long time. If any of the child process is still running when\n        `max_run_time` is up, they will be force-terminated and an\n        `UnexpectedSubprocessExitError` may be raised. If `None`, child\n        processes are not forced to exit.\n      grpc_fail_fast: Whether GRPC connection between processes should fail\n        without retrying. Defaults to None, in which case the environment\n        variable is not explicitly set.\n      stream_output: True if the output/error from the subprocesses should be\n        streamed to be printed in parent process\' log. Defaults to True.\n      return_output: If True, the output/error from the subprocesses should be\n        collected to be attached to the resulting namedtuple returned from\n        `join()`. The list of output can be retrieved via `stdout` attribute.\n        Defaults to False.\n      use_dill_for_args: Whether to use dill to pickle `args` and `kwargs`. dill\n        can pickle more objects, but doesn\'t work with types in\n        `multiprocessing` library like `Mutex`.\n      daemon: Whether to start processes as daemons.\n      dependence_on_chief: Whether to terminates the cluster if the chief exits.\n        If auto_restart is True, it only terminates the cluster if the chief\n        exits with a zero exit code.\n      auto_restart: Whether to automatically restart processes that exit with\n        non-zero exit code.\n      share_gpu: Whether to share GPUs among workers. If False, each worker is\n        assigned different GPUs in a roundrobin fashion. This should be True\n        whenever possible for better test execution coverage; some situations\n        that need it to be False are tests that runs NCCL.\n      args: Positional arguments to be sent to `fn` run on subprocesses.\n      kwargs: Keyword arguments to be sent to `fn` run on subprocesses.\n\n    Raises:\n      RuntimeError: if `multi_process_runner.test_main()` is not called.\n      ValueError: if there are more than one chief in the `cluster_spec`.\n      SkipTest: if thread sanitizer is enabled (which is incompatible with MPR).\n    '
        if test_util.is_tsan_enabled():
            raise unittest.SkipTest('ThreadSanitizer is not compatible with MultiProcessRunner.')
        assert cluster_spec is not None
        if 'chief' in cluster_spec and len(cluster_spec['chief']) > 1:
            raise ValueError('If chief exists in the cluster, there must be at most one chief. Current `cluster_spec` has {} chiefs.'.format(len(cluster_spec['chief'])))
        _check_initialization()
        if not callable(fn):
            raise ValueError('fn is not a callable')
        self._fn = fn
        self._cluster_spec = cluster_spec
        self._rpc_layer = rpc_layer or 'grpc'
        self._max_run_time = max_run_time
        self._grpc_fail_fast = grpc_fail_fast
        self._stream_output = stream_output
        self._return_output = return_output
        self._dependence_on_chief = dependence_on_chief
        self._use_dill_for_args = use_dill_for_args
        self._daemon = daemon
        self._auto_restart = auto_restart
        self._args = args or ()
        self._kwargs = kwargs or {}
        self._share_gpu = share_gpu
        self._total_gpu = len(context.context().list_physical_devices('GPU'))
        self._v2_enabled = tf2.enabled()
        self._executing_eagerly = context.executing_eagerly()
        self._joined = False
        self._process_lock = threading.Lock()
        self._processes = {}
        self._terminated = set()
        self._reading_threads = []
        self._manager = manager()
        self._process_status_queue = self._manager.Queue()
        self._parent_to_sub_queue = self._manager.Queue()
        parties = sum((len(addresses) for addresses in self._cluster_spec.values()))
        self._barrier = self._manager.Barrier(parties)
        self._streaming_queue = self._manager.Queue()
        self._watchdog_thread = None

    def set_args(self, args=None, kwargs=None):
        if False:
            print('Hello World!')
        self._args = args or self._args
        self._kwargs = kwargs or self._kwargs

    def _continuously_readline_from_sub(self, pipe_r, task_type, task_id):
        if False:
            return 10
        'Function to continuously read lines from subprocesses.'
        with os.fdopen(pipe_r.fileno(), 'r', closefd=False) as reader:
            for line in reader:
                task_string = '[{}-{}]:'.format(task_type, task_id)
                formatted_line = '{} {}'.format(task_string.ljust(14), line)
                if self._stream_output:
                    print(formatted_line, end='', flush=True)
                if self._return_output:
                    self._streaming_queue.put(formatted_line)

    def _start_subprocess_and_reading_thread(self, task_type, task_id, cluster_spec=None, fn=None, args=None, kwargs=None):
        if False:
            return 10
        'Start a subprocess and a thread the reads lines from the subprocess.'
        if dill is None:
            raise unittest.SkipTest('TODO(b/150264776): Resolve dependency issue in CI')
        cluster_spec = cluster_spec or self._cluster_spec
        visible_gpus = None
        if not self._share_gpu and self._total_gpu > 0:
            id_in_cluster = multi_worker_util.id_in_cluster(cluster_spec, task_type, task_id)
            worker_count = multi_worker_util.worker_count(cluster_spec, task_type)
            visible_gpus = list(range(id_in_cluster, self._total_gpu, worker_count))
        test_env = TestEnvironment(task_type=task_type, task_id=task_id, cluster_spec=cluster_spec, rpc_layer=self._rpc_layer, grpc_fail_fast=self._grpc_fail_fast, v2_enabled=self._v2_enabled, executing_eagerly=self._executing_eagerly, visible_gpus=visible_gpus)
        (pipe_r, pipe_w) = multiprocessing.Pipe(duplex=False)
        resources = Resources(process_status_queue=self._process_status_queue, parent_to_sub_queue=self._parent_to_sub_queue, streaming_pipe_w=pipe_w, barrier=self._barrier)
        if fn is None:
            (fn, args, kwargs) = (self._fn, self._args, self._kwargs)
        fn = dill.dumps(fn, dill.HIGHEST_PROTOCOL)
        if self._use_dill_for_args:
            args = dill.dumps(args, dill.HIGHEST_PROTOCOL)
            kwargs = dill.dumps(kwargs, dill.HIGHEST_PROTOCOL)
        p = _Process(test_env=test_env, target=_ProcFunc(), args=(resources, test_env, fn, args, kwargs, self._use_dill_for_args), daemon=self._daemon)
        p.start()
        self._processes[task_type, task_id] = p
        self._terminated.discard((task_type, task_id))
        thread = threading.Thread(target=self._continuously_readline_from_sub, args=(pipe_r, task_type, task_id))
        thread.start()
        self._reading_threads.append(thread)
        if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
            self._watchdog_thread = threading.Thread(target=self._process_watchdog)
            self._watchdog_thread.start()

    def start(self):
        if False:
            return 10
        'Starts processes, one for each task in `cluster_spec`.\n\n    Note that this is best effort by the applicable multiprocessing library,\n    and it may take up to seconds for a subprocess to be successfully started.\n    '
        with self._process_lock:
            if self._processes:
                raise ValueError('MultiProcessRunner already started.')
            if self._joined:
                raise ValueError('cannot start new processes afterMultiProcessRunner.join() is called')
            for (task_type, addresses) in self._cluster_spec.items():
                for (task_id, _) in enumerate(addresses):
                    self._start_subprocess_and_reading_thread(task_type, task_id)
        if self._max_run_time is not None:

            def handler(signum, frame):
                if False:
                    i = 10
                    return i + 15
                del signum, frame
                self.terminate_all()
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(self._max_run_time)

    def start_in_process_as(self, as_task_type, as_task_id):
        if False:
            print('Hello World!')
        "Start the processes, with the specified task run in main process.\n\n    This is similar to `start()` except that the task with task_type\n    `as_task_type` and task_id `as_task_id` is run in the main process.\n    This method is particularly useful when debugging tool such as `pdb` is\n    needed in some specific task. Note that since this method is blocking until\n    that specific task exits, additional actions would need a thread to be\n    called:\n\n    ```python\n    def fn():\n      # user code to be run\n      import pdb; pdb.set_trace()\n\n    def follow_ups():\n      time.sleep(5)\n      mpr.start_single_process(\n          task_type='evaluator',\n          task_id=0)\n\n    mpr = multi_process_runner.MultiProcessRunner(\n        fn,\n        multi_worker_test_base.create_cluster_spec(\n            has_chief=True, num_workers=1))\n    threading.Thread(target=follow_ups).start()\n    mpr.start_in_process_as(as_task_type='chief', as_task_id=0)\n    mpr.join()\n    ```\n\n    Note that if `return_output=True`, the logs/stdout by task\n    run by the main process is not available in result.stdout.\n\n    Args:\n      as_task_type: The task type to be run in the main process.\n      as_task_id: The task id to be run in the main process.\n    "
        if self._processes:
            raise ValueError('MultiProcessRunner already started.')
        with self._process_lock:
            if self._joined:
                raise ValueError('cannot start new processes afterMultiProcessRunner.join() is called')
            for (task_type, addresses) in self._cluster_spec.items():
                for (task_id, _) in enumerate(addresses):
                    if not (task_type == as_task_type and task_id == as_task_id):
                        self._start_subprocess_and_reading_thread(task_type, task_id)
        _set_tf_config(as_task_type, as_task_id, self._cluster_spec, self._rpc_layer)
        self._fn(*self._args, **self._kwargs)

    def start_single_process(self, task_type, task_id, cluster_spec=None, fn=None, args=None, kwargs=None):
        if False:
            print('Hello World!')
        'Starts a single process.\n\n    This starts a process in the cluster with the task type, task id, and the\n    process function (`fn`). If process function is `None`, the function\n    provided at `__init__` will be used. If `cluster_spec` is `None`, the\n    cluster spec provided at `__init__` will be used.\n\n    TODO(rchao): It is meant that all subprocesses will be updated with the new\n    cluster spec, but this has yet to be implemented. At this time only the\n    newly started subprocess picks up this updated cluster spec.\n\n    Args:\n      task_type: The task type.\n      task_id: The task id.\n      cluster_spec: The cluster spec to be used on the newly started\n        process. If `None`, the cluster spec provided at `__init__` will be\n        used.\n      fn: The process function to be run on the newly started\n        process. If specified, specify `args` and `kwargs` as well. If `None`,\n        the function provided at `__init__` will be used.\n      args: Optional positional arguments to be supplied in `fn`.\n      kwargs: Optional keyword arguments to be supplied in `fn`.\n    '
        with self._process_lock:
            if self._joined:
                raise ValueError('cannot start new processes afterMultiProcessRunner.join() is called')
            self._start_subprocess_and_reading_thread(task_type, task_id, cluster_spec=cluster_spec, fn=fn, args=args or (), kwargs=kwargs or {})

    def _queue_to_list(self, queue_to_convert):
        if False:
            while True:
                i = 10
        'Convert `queue.Queue` to `list`.'
        list_to_return = []
        while True:
            try:
                list_to_return.append(queue_to_convert.get(block=False))
            except Queue.Empty:
                break
        return list_to_return

    def _get_process_statuses(self):
        if False:
            while True:
                i = 10
        statuses = {}
        for status in self._queue_to_list(self._process_status_queue):
            statuses[status.task_type, status.task_id] = status
        return statuses

    def get_process_id(self, task_type, task_id):
        if False:
            for i in range(10):
                print('nop')
        'Returns the subprocess id given the task type and task id.'
        with self._process_lock:
            p = self._processes.get((task_type, task_id), None)
        return p.pid if p else None

    def get_process_exit_code(self, task_type, task_id):
        if False:
            while True:
                i = 10
        'Returns the subprocess exit code given the task type and task id.\n\n    Args:\n      task_type: The task type.\n      task_id: The task id.\n\n    Returns:\n      The subprocess exit code; `None` if the subprocess has not exited yet.\n\n    Raises:\n      KeyError: If the corresponding subprocess is not found with `task_type`\n        and `task_id`.\n    '
        with self._process_lock:
            p = self._processes[task_type, task_id]
        return p.exitcode if p else None

    def process_exists(self, task_type, task_id):
        if False:
            for i in range(10):
                print('nop')
        'Returns whether the subprocess still exists given the task type and id.\n\n    Args:\n      task_type: The task type.\n      task_id: The task id.\n\n    Returns:\n      Boolean; whether the subprocess still exists. If the subprocess has\n      exited, this returns False.\n    '
        return self.get_process_exit_code(task_type, task_id) is None

    def _process_watchdog(self):
        if False:
            for i in range(10):
                print('nop')
        'Simulates a cluster management system.\n\n    - If auto_restart is True, it restarts processes that exit with a non-zero\n      exit code. Note that when join() times out it overrides auto_restart to\n      False.\n    - If dependence_on_chief is True, it terminates all processes once the chief\n      exits. If auto_restart is also True, it only terminates all processes if\n      the chief exit with a zero exit code, otherwise it restarts the chief.\n\n    This runs in self._watchdog_thread.\n    '
        while True:
            time.sleep(1)
            with self._process_lock:
                chief = self._processes.get(('chief', 0), None)
                if chief and self._dependence_on_chief and (chief.exitcode is not None):
                    if chief.exitcode == 0 or not self._auto_restart:
                        for p in self._processes.values():
                            p.join(timeout=3)
                        self._terminate_all()
                        for p in self._processes.values():
                            p.join()
                        return
                if self._auto_restart:
                    has_failure = False
                    for ((task_type, task_id), p) in self._processes.items():
                        if p.exitcode is not None and p.exitcode != 0:
                            has_failure = True
                            logging.info('Restarting failed %s-%d', task_type, task_id)
                            self._start_subprocess_and_reading_thread(task_type, task_id)
                    if has_failure:
                        continue
                if all((p.exitcode is not None for p in self._processes.values())):
                    return

    def _reraise_if_subprocess_error(self, process_statuses):
        if False:
            return 10
        for process_status in process_statuses.values():
            assert isinstance(process_status, _ProcessStatusInfo)
            if not process_status.is_successful:
                process_status.exc_info[1].mpr_result = self._get_mpr_result(process_statuses)
                six.reraise(*process_status.exc_info)

    def join(self, timeout=_DEFAULT_TIMEOUT_SEC):
        if False:
            for i in range(10):
                print('nop')
        "Joins all the processes with timeout.\n\n    If any of the subprocesses does not exit approximately after `timeout`\n    seconds has passed after `join` call, this raises a\n    `SubprocessTimeoutError`.\n\n    Note: At timeout, it uses SIGTERM to terminate the subprocesses, in order to\n    log the stack traces of the subprocesses when they exit. However, this\n    results in timeout when the test runs with tsan (thread sanitizer); if tsan\n    is being run on the test targets that rely on timeout to assert information,\n    `MultiProcessRunner.terminate_all()` must be called after `join()`, before\n    the test exits, so the subprocesses are terminated with SIGKILL, and data\n    race is removed.\n\n    Args:\n      timeout: optional integer or `None`. If provided as an integer, and not\n      all processes report status within roughly `timeout` seconds, a\n      `SubprocessTimeoutError` exception will be raised. If `None`, `join` never\n      times out.\n\n    Returns:\n      A `MultiProcessRunnerResult` object, which has two attributes,\n      `return_value` and `stdout`. `return_value` always contains a list of\n      return values from the subprocesses, although the order is not meaningful.\n      If `return_output` argument is True at `__init__`, `stdout` is available\n      that contains a list of all messages from subprocesses' stdout and stderr.\n\n    Raises:\n      SubprocessTimeoutError: if not all processes report status approximately\n        within `timeout` seconds. When this is raised, a\n        `MultiProcessRunnerResult` object can be retrieved by\n        `SubprocessTimeoutError`'s mpr_result attribute, which has the same\n        structure as above 'Returns' section describes.\n      UnexpectedSubprocessExitError: If any of the subprocesses did not exit\n        properly (for example, they exit on SIGTERM or SIGKILL signal). When\n        this is raised, a `MultiProcessRunnerResult` object can be retrieved by\n        `UnexpectedSubprocessExitError`'s mpr_result attribute, which has the\n        same structure as above 'Returns' section describes. If `max_run_time`\n        is not `None`, it is expected that some subprocesses may be\n        force-killed when `max_run_time` is up, and this is raised in those\n        cases.\n      Exception: if there is an Exception propagated from any subprocess. When\n        this is raised, a `MultiProcessRunnerResult` object can be retrieved by\n        `UnexpectedSubprocessExitError`'s mpr_result attribute, which has the\n        same structure as above 'Returns' section describes.\n    "
        if timeout and (not isinstance(timeout, int)):
            raise ValueError('`timeout` must be an integer or `None`.')
        with self._process_lock:
            if self._joined:
                raise ValueError("MultiProcessRunner can't be joined twice.")
            self._joined = True
        self._watchdog_thread.join(timeout)
        if self._watchdog_thread.is_alive():
            with self._process_lock:
                self._auto_restart = False
            logging.error('Timeout when joining for child processes. Terminating...')
            self.terminate_all(sig=signal.SIGTERM)
            self._watchdog_thread.join(_FORCE_KILL_WAIT_SEC)
            if self._watchdog_thread.is_alive():
                logging.error('Timeout when waiting for child processes to print stacktrace. Sending SIGKILL...')
                self.terminate_all()
                self._watchdog_thread.join()
            process_statuses = self._get_process_statuses()
            self._reraise_if_subprocess_error(process_statuses)
            raise SubprocessTimeoutError('One or more subprocesses timed out, where timeout was set to {}s. Please change the `timeout` argument for `MultiProcessRunner.join()` or `multi_process_runner.run()` if it should be adjusted.'.format(timeout), self._get_mpr_result(process_statuses))
        for ((task_type, task_id), p) in self._processes.items():
            logging.info('%s-%d exit code: %s', task_type, task_id, p.exitcode)
        process_statuses = self._get_process_statuses()
        self._reraise_if_subprocess_error(process_statuses)
        for ((task_type, task_id), p) in self._processes.items():
            assert p.exitcode is not None
            if p.exitcode > 0 and (task_type, task_id) not in self._terminated:
                raise UnexpectedSubprocessExitError('Subprocess %s-%d exited with exit code %s. See logs for details.' % (task_type, task_id, p.exitcode), self._get_mpr_result(process_statuses))
        logging.info('Joining log reading threads.')
        for thread in self._reading_threads:
            thread.join()
        logging.info('Joined log reading threads.')
        signal.alarm(0)
        return self._get_mpr_result(process_statuses)

    def _get_mpr_result(self, process_statuses):
        if False:
            while True:
                i = 10
        stdout = self._queue_to_list(self._streaming_queue)
        return_values = []
        for process_status in process_statuses.values():
            if process_status.return_value is not None:
                return_values.append(process_status.return_value)
        return MultiProcessRunnerResult(stdout=stdout, return_value=return_values)

    def terminate(self, task_type, task_id):
        if False:
            i = 10
            return i + 15
        'Terminates the process with `task_type` and `task_id`.\n\n    If auto_retart=True, the terminated task will be restarted unless the chief\n    has already exited with zero exit code.\n\n    Args:\n      task_type: the task type.\n      task_id: the task id.\n\n    '
        with self._process_lock:
            p = self._processes.get((task_type, task_id), None)
            if p is None:
                raise ValueError('{}-{} does not exist'.format(task_type, task_id))
            self._terminated.add((task_type, task_id))
            self._parent_to_sub_queue.put('terminate {} {}'.format(task_type, task_id))
            p.join()

    def _terminate_all(self, sig=None):
        if False:
            i = 10
            return i + 15
        'Terminates all subprocesses.\n\n    The caller is required to hold self._process_lock.\n\n    Args:\n      sig: the signal used to terminate the process. The default is SIGKILL.\n    '
        sig = sig or getattr(signal, 'SIGKILL', signal.SIGTERM)
        for ((task_type, task_id), p) in self._processes.items():
            if p.exitcode is not None:
                logging.info('%s-%d has already exited. Not terminating.', task_type, task_id)
                continue
            try:
                os.kill(p.pid, sig)
                self._terminated.add((task_type, task_id))
                logging.info('%s-%d terminated with signal %r.', task_type, task_id, sig)
            except ProcessLookupError:
                logging.info('Attempting to kill %s-%d but it does not exist.', task_type, task_id)

    def terminate_all(self, sig=None):
        if False:
            return 10
        'Terminates all subprocesses.'
        with self._process_lock:
            self._terminate_all(sig)

class _Process(multi_process_lib.Process):
    """A modified `multiprocessing.Process` that can set up environment variables."""

    def __init__(self, test_env, **kwargs):
        if False:
            print('Hello World!')
        super(_Process, self).__init__(**kwargs)
        self._test_env = test_env
        self._actual_run = getattr(self, 'run')
        self.run = self._run_with_setenv

    def _run_with_setenv(self):
        if False:
            for i in range(10):
                print('nop')
        test_env = self._test_env
        if test_env.grpc_fail_fast is not None:
            os.environ['GRPC_FAIL_FAST'] = str(test_env.grpc_fail_fast)
        if test_env.visible_gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in test_env.visible_gpus])
        _set_tf_config(test_env.task_type, test_env.task_id, test_env.cluster_spec, test_env.rpc_layer)
        return self._actual_run()

class _ProcFunc(object):
    """Represents a callable to run in a subprocess."""

    @contextlib.contextmanager
    def _runtime_mode(self, executing_eagerly):
        if False:
            i = 10
            return i + 15
        if executing_eagerly:
            with context.eager_mode():
                yield
        else:
            with context.graph_mode():
                yield

    def _message_checking_func(self, task_type, task_id):
        if False:
            while True:
                i = 10
        'A function that regularly checks messages from parent process.'
        while True:
            try:
                message = self._resources.parent_to_sub_queue.get(block=False)
                if not message.startswith('terminate'):
                    raise ValueError('Unrecognized message: {}'.format(message))
                if message == 'terminate {} {}'.format(task_type, task_id):
                    break
                else:
                    self._resources.parent_to_sub_queue.put(message)
                    time.sleep(1)
            except Queue.Empty:
                time.sleep(0.1)
        self._resources.process_status_queue.put(_ProcessStatusInfo(task_type=task_type, task_id=task_id, is_successful=True, exc_info=None, return_value=None))
        os._exit(1)

    def _close_streaming(self):
        if False:
            return 10
        'Close stdout, stderr and streaming pipe.\n\n    We need to explicitly close them since Tensorflow may take a while to exit,\n    so that the reading threads in the main process can exit more quickly.\n    '
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout.close()
        sys.stderr.close()
        self._resources.streaming_pipe_w.close()

    def __call__(self, resources, test_env, fn, args, kwargs, use_dill_for_args):
        if False:
            for i in range(10):
                print('nop')
        'The wrapper function that actually gets run in child process(es).'
        global _barrier
        self._resources = resources
        _barrier = self._resources.barrier
        fn = dill.loads(fn)
        if use_dill_for_args:
            args = dill.loads(args)
            kwargs = dill.loads(kwargs)
        if faulthandler is not None:
            faulthandler.enable()
            faulthandler.register(signal.SIGTERM, chain=True)
        logging.set_stderrthreshold(logging.DEBUG)
        os.dup2(resources.streaming_pipe_w.fileno(), sys.stdout.fileno())
        os.dup2(resources.streaming_pipe_w.fileno(), sys.stderr.fileno())
        pid = os.getpid()
        logging.info('Subprocess with PID %d (%s, %d) is now being started.', pid, test_env.task_type, test_env.task_id)
        logging.info('TF_CONFIG: %r', os.environ['TF_CONFIG'])
        threading.Thread(target=self._message_checking_func, args=(test_env.task_type, test_env.task_id), daemon=True).start()
        if test_env.v2_enabled:
            v2_compat.enable_v2_behavior()
        with self._runtime_mode(test_env.executing_eagerly):
            info = _run_contained(test_env.task_type, test_env.task_id, fn, args, kwargs)
            self._resources.process_status_queue.put(info)
            if not info.is_successful:
                six.reraise(*info.exc_info)
            self._close_streaming()
        sys.exit(0)
_active_pool_runners = weakref.WeakSet()

def _shutdown_all_pool_runners():
    if False:
        return 10
    for pool in _active_pool_runners:
        pool.shutdown()

def is_oss():
    if False:
        while True:
            i = 10
    'Returns whether the test is run under OSS.'
    return len(sys.argv) >= 1 and 'bazel' in sys.argv[0]

class MultiProcessPoolRunner(object):
    """A utility class to start a process pool to simulate a cluster.

  It's similar to MultiProcessRunner, but uses a pool of processes to avoid the
  expensive initialization cost of Tensorflow.
  """

    def __init__(self, cluster_spec, initializer=None, share_gpu=True):
        if False:
            return 10
        'Creates a multi-process pool runner.\n\n    Args:\n      cluster_spec: Dict for cluster spec. The following is an example of\n        cluster with three workers.\n        {"worker": ["worker0.example.com:2222",\n                    "worker1.example.com:2222",\n                    "worker2.example.com:2222"]}\n      initializer: a callable to called at the startup of worker processes.\n      share_gpu: Whether to share GPUs among workers. If False, each worker is\n        assigned different GPUs in a roundrobin fashion.\n\n    Raises:\n      RuntimeError: if `multi_process_runner.test_main()` is not called.\n      ValueError: if there are more than one chief in the `cluster_spec`.\n    '
        _active_pool_runners.add(self)
        self._cluster_spec = cluster_spec
        self._initializer = initializer
        self._share_gpu = share_gpu
        self._conn = {}
        self._runner = None

    def __del__(self):
        if False:
            return 10
        self.shutdown()

    def shutdown(self):
        if False:
            print('Hello World!')
        'Shuts down the worker pool.'
        for conn in self._conn.values():
            conn.close()
        self._conn = {}
        if self._runner is not None:
            try:
                self._runner.join()
            except Exception as e:
                logging.error('Ignoring exception when shutting down MultiProcessPoolRunner: %s', e)
            self._runner = None

    def _start(self):
        if False:
            i = 10
            return i + 15
        'Starts the worker pool.'
        if dill is None:
            raise unittest.SkipTest('TODO(b/150264776): Resolve dependency issue in CI')
        self._runner = MultiProcessRunner(fn=lambda : None, cluster_spec=self._cluster_spec, use_dill_for_args=False, share_gpu=self._share_gpu)
        if self._initializer:
            initializer = dill.dumps(self._initializer, dill.HIGHEST_PROTOCOL)
        else:
            initializer = None
        for (task_type, addresses) in self._cluster_spec.items():
            for (task_id, _) in enumerate(addresses):
                (conn1, conn2) = multiprocessing.Pipe(duplex=True)
                self._conn[task_type, task_id] = conn1
                self._runner.start_single_process(task_type, task_id, fn=_pool_runner_worker, args=(task_type, task_id, initializer, conn2))

    def run(self, fn, args=None, kwargs=None):
        if False:
            print('Hello World!')
        'Runs `fn` with `args` and `kwargs` on all jobs.\n\n    Args:\n      fn: The function to be run.\n      args: Optional positional arguments to be supplied in `fn`.\n      kwargs: Optional keyword arguments to be supplied in `fn`.\n\n    Returns:\n      A list of return values.\n    '
        _check_initialization()
        multi_process_lib.Process()
        if self._runner is None:
            self._start()
        fn = dill.dumps(fn, dill.HIGHEST_PROTOCOL)
        for conn in self._conn.values():
            conn.send((fn, args or [], kwargs or {}))
        process_statuses = []
        for ((task_type, task_id), conn) in self._conn.items():
            logging.info('Waiting for the result from %s-%d', task_type, task_id)
            try:
                process_statuses.append(conn.recv())
            except EOFError:
                self.shutdown()
                raise RuntimeError('Unexpected EOF. Worker process may have died. Please report a bug')
        return_values = []
        for process_status in process_statuses:
            assert isinstance(process_status, _ProcessStatusInfo)
            if not process_status.is_successful:
                six.reraise(*process_status.exc_info)
            if process_status.return_value is not None:
                return_values.append(process_status.return_value)
        return return_values

def _pool_runner_worker(task_type, task_id, initializer, conn):
    if False:
        i = 10
        return i + 15
    'Function that runs on the workers in a pool.\n\n  It listens for callables to run and returns the result until `conn` is closed.\n  It captures the exceptions during executing the callable and return it through\n  `conn`.\n\n  Args:\n    task_type: the task type.\n    task_id: the task index.\n    initializer: a callable to execute during startup.\n    conn: a multiprocessing.Connection object to listen for tasks and send\n      results.\n  '
    if initializer:
        initializer = dill.loads(initializer)
        initializer()
    while True:
        try:
            (fn, args, kwargs) = conn.recv()
        except EOFError:
            break
        fn = dill.loads(fn)
        info = _run_contained(task_type, task_id, fn, args, kwargs)
        sys.stdout.flush()
        sys.stderr.flush()
        conn.send(info)

def _run_contained(task_type, task_id, fn, args, kwargs):
    if False:
        print('Hello World!')
    'Runs `fn` with `args` and `kwargs`.\n\n  The function returns _ProcessStatusInfo which captures the return value and\n  the exception.\n\n  Args:\n    task_type: the task type.\n    task_id: the task index.\n    fn: the function to be run.\n    args: optional positional arguments to be supplied in `fn`.\n    kwargs: optional keyword arguments to be supplied in `fn`.\n\n  Returns:\n    a _ProcessStatusInfo.\n\n  '
    is_successful = False
    return_value = None
    exc_info = None
    try:
        return_value = fn(*args, **kwargs)
        is_successful = True
        return _ProcessStatusInfo(task_type=task_type, task_id=task_id, is_successful=is_successful, exc_info=exc_info, return_value=return_value)
    except Exception:
        exc_info = sys.exc_info()
        return _ProcessStatusInfo(task_type=task_type, task_id=task_id, is_successful=is_successful, exc_info=exc_info, return_value=return_value)

@tf_export('__internal__.distribute.multi_process_runner.SubprocessTimeoutError', v1=[])
class SubprocessTimeoutError(RuntimeError):
    """An error that indicates there is at least one subprocess timing out.

  When this is raised, a namedtuple object representing the multi-process run
  result can be retrieved by
  `tf.__internal__.distribute.multi_process_runner.SubprocessTimeoutError`'s
  `mpr_result` attribute. See
  `tf.__internal__.distribute.multi_process_runner.run` for more information.
  """

    def __init__(self, msg, mpr_result):
        if False:
            i = 10
            return i + 15
        super(SubprocessTimeoutError, self).__init__(msg)
        self.mpr_result = mpr_result

@tf_export('__internal__.distribute.multi_process_runner.UnexpectedSubprocessExitError', v1=[])
class UnexpectedSubprocessExitError(RuntimeError):
    """An error indicating there is at least one subprocess with unexpected exit.

  When this is raised, a namedtuple object representing the multi-process run
  result can be retrieved by
  `tf.__internal__.distribute.multi_process_runner
  .UnexpectedSubprocessExitError`'s
  `mpr_result` attribute. See
  `tf.__internal__.distribute.multi_process_runner.run` for more information.
  """

    def __init__(self, msg, mpr_result):
        if False:
            print('Hello World!')
        super(UnexpectedSubprocessExitError, self).__init__(msg)
        self.mpr_result = mpr_result

@tf_export('__internal__.distribute.multi_process_runner.NotInitializedError', v1=[])
class NotInitializedError(RuntimeError):
    """An error indicating `multi_process_runner.run` is used without init.

  When this is raised, user is supposed to call
  `tf.__internal__.distribute.multi_process_runner.test_main()` within
  `if __name__ == '__main__':` block to properly initialize
  `multi_process_runner.run`.
  """
    pass

def _check_initialization():
    if False:
        return 10
    if not multi_process_lib.initialized():
        raise NotInitializedError("`multi_process_runner` is not initialized. Please call `tf.__internal__.distribute.multi_process_runner.test_main()` within `if __name__ == '__main__':` block in your python module to properly initialize `multi_process_runner`.")

def _set_tf_config(task_type, task_id, cluster_spec, rpc_layer=None):
    if False:
        return 10
    'Set TF_CONFIG environment variable.'
    tf_config_dict = {'cluster': cluster_spec, 'task': {'type': task_type, 'index': task_id}}
    if rpc_layer is not None:
        tf_config_dict['rpc_layer'] = rpc_layer
    os.environ['TF_CONFIG'] = json.dumps(tf_config_dict)

@tf_export('__internal__.distribute.multi_process_runner.run', v1=[])
def run(fn, cluster_spec, rpc_layer=None, max_run_time=None, return_output=False, timeout=_DEFAULT_TIMEOUT_SEC, args=None, kwargs=None):
    if False:
        print('Hello World!')
    'Run `fn` in multiple processes according to `cluster_spec`.\n\n  Given a callable `fn`, `tf.__internal__.distribute.multi_process_runner.run`\n  launches multiple processes, each of which runs `fn`. These processes are\n  referred to as "subprocesses" or "child processes". Each of those subprocesses\n  will have their `TF_CONFIG` environment variable set, according to\n  `cluster_spec` and their task types. The stdout of the subprocesses are\n  streamed to the main process\' and thus available in logs (if `stream_output`\n  is True), with [type-id] prefix.\n\n  `tf.__internal__.distribute.multi_process_runner.run` will block until all\n  subprocesses have successfully exited, and return a namedtuple object that\n  represents the run result. This object has a `return_value` attribute, which\n  is a list that contains subprocesses `fn`\'s return values, for those\n  subprocesses that successfully returned from `fn`. The order of `return_value`\n  list is not meaningful. If an optional arg `return_output` (default to False)\n  is set to True, the namedtuple object will have an additional attribute\n  `stdout`, which is a list containing the stdout of the subprocesses. If any\n  subprocess\' `fn` ends up raising an error, that error will be reraised from\n  `tf.__internal__.distribute.multi_process_runner.run`, and the aforementioned\n  namedtuple object will be available through the exception\'s\n  `mpr_result` attribute.\n\n  This utility is used for simulating running TensorFlow programs across\n  multiple task types, and each of the task type may contain more than one task\n  (except for "chief" where more than one task is prohibited). Test coverage of\n  multi-worker training is the main application of this utility, where code\n  written for multi-worker training can be realistically covered in unit tests.\n\n  Any test module that uses\n  `tf.__internal__.distribute.multi_process_runner.run()` must call\n  `tf.__internal__.distribute.multi_process_runner.test_main()` instead of\n  regular `test.main()` inside `if __name__ == \'__main__\':` block for proper\n  initialization.\n\n  Args:\n    fn: Function to be run on child processes. This will be run on processes for\n      all task types.\n    cluster_spec: Dict for cluster spec. The utility function\n      `tf.__internal__.distribute.multi_process_runner.create_cluster_spec` can\n      be conveniently used to create such dict. The following is an example of\n      cluster with three workers and two ps\'s.\n      {"worker": ["worker0.example.com:2222",\n                  "worker1.example.com:2222",\n                  "worker2.example.com:2222"],\n       "ps": ["ps0.example.com:2222",\n              "ps1.example.com:2222"]}\n    rpc_layer: RPC layer to use. Default value is \'grpc\'.\n    max_run_time: `None` or integer. If not `None`, child processes are forced\n      to exit at approximately this many seconds after this utility is called.\n      We achieve this through `signal.alarm()` api. Note that this is best\n      effort at Python level since Python signal handler does not get executed\n      when it runs lower level C/C++ code. So it can be delayed for arbitrarily\n      long time. If any of the child process is still running when\n      `max_run_time` is up, they will be force-terminated and an\n      `tf.__internal__.distribute.multi_process_runner\n      .UnexpectedSubprocessExitError`\n      may be raised. If `None`, child processes are not forced to exit.\n    return_output: If True, the output/error from the subprocesses should be\n      collected to be attached to the resulting namedtuple returned from this\n      utility. The list of output can be retrieved via `stdout` attribute.\n      Defaults to False.\n    timeout: optional integer or `None`. If provided as an integer, and not all\n      processes report status within roughly `timeout` seconds, a\n      `tf.__internal__.distribute.multi_process_runner.SubprocessTimeoutError`\n      exception will be raised. If `None`,\n      `tf.__internal__.distribute.multi_process_runner.run` never times out.\n      Defaults to the constant `_DEFAULT_TIMEOUT_SEC` defined in\n      `multi_process_runner` module.\n    args: Positional arguments to be sent to `fn` run on subprocesses.\n    kwargs: Keyword arguments to be sent to `fn` run on subprocesses.\n\n  Returns:\n      A namedtuple object, which has two attributes,\n      `return_value` and `stdout`. `return_value` always contains a list of\n      returnvalues from the subprocesses, although the order is not meaningful.\n      If `return_output` argument is True, `stdout` is available that contains a\n      list of all messages from subprocesses\' stdout and stderr, and the order\n      is mostly chronological.\n\n  Raises:\n    RuntimeError: if\n    `tf.__internal__.distribute.multi_process_runner.test_main()` is\n      not called in test\'s `if __name__ == \'__main__\':` block.\n    ValueError: if there are more than one chief in the `cluster_spec`.\n    tf.__internal__.distribute.multi_process_runner.SubprocessTimeoutError: if\n      not all processes report status approximately\n      within `timeout` seconds. When this is raised, a\n      namedtuple object can be retrieved by\n      `tf.__internal__.distribute.multi_process_runner.SubprocessTimeoutError`\'s\n      `mpr_result` attribute, which has the same\n      structure as above \'Returns\' section describes.\n    tf.__internal__.distribute.multi_process_runner\n    .UnexpectedSubprocessExitError:\n      If any of the subprocesses did not exit\n      properly (for example, they exit on SIGTERM or SIGKILL signal). When\n      this is raised, a namedtuple object can be retrieved by\n      `tf.__internal__.distribute.multi_process_runner\n      .UnexpectedSubprocessExitError`\'s\n      `mpr_result` attribute, which has the\n      same structure as above \'Returns\' section describes. If `max_run_time`\n      is not `None`, it is expected that some subprocesses may be\n      force-killed when `max_run_time` is up, and this is raised in those\n      cases.\n    Exception: if there is an Exception propagated from any subprocess. When\n      this is raised, a namedtuple object can be retrieved by\n      `tf.__internal__.distribute.multi_process_runner\n      .UnexpectedSubprocessExitError`\n      `mpr_result` attribute, which has the\n      same structure as above \'Returns\' section describes.\n\n  Examples:\n\n  ```python\n  class SimpleMultiProcessTest(tf.test.TestCase):\n\n    def test_simple_printing_and_return(self):\n\n      def fn():\n        resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()\n\n        # This will print "[chief-0]:     Task type: chief , task id: 0"\n        # for chief, for example.\n        logging.info(\'Task type: %s, task id: %d\',\n                     resolver.task_type, resolver.task_id)\n\n        return resolver.task_type\n\n      result = tf.__internal__.distribute.multi_process_runner.run(\n          fn=fn,\n          cluster_spec=(\n              tf.__internal__\n              .distribute.multi_process_runner.create_cluster_spec(\n                  has_chief=True, num_workers=2)))\n      assert sorted(result.return_value) == [\'chief\', \'worker\', \'worker\']\n\n    def test_error_from_fn(self):\n\n      def fn():\n        resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()\n        raise ValueError(\'Task type {}, task id {} is errors out\'.format(\n            resolver.task_type, resolver.task_id))\n\n      with self.assertRaisesRegexp(ValueError,\n                                   \'Task type worker, task id 0 is errors out\'):\n        cluster_spec = (\n            tf.__internal__.distribute.multi_process_runner.create_cluster_spec(\n                num_workers=1))\n        tf.__internal__.distribute.multi_process_runner.run(\n            fn=fn, cluster_spec=cluster_spec)\n\n\n  if __name__ == \'__main__\':\n    tf.__internal__.distribute.multi_process_runner.test_main()\n  ```\n  '
    runner = MultiProcessRunner(fn, cluster_spec, rpc_layer, max_run_time=max_run_time, return_output=return_output, args=args, kwargs=kwargs)
    runner.start()
    return runner.join(timeout)
_barrier = None

@tf_export('__internal__.distribute.multi_process_runner.get_barrier', v1=[])
def get_barrier():
    if False:
        while True:
            i = 10
    'Returns a `multiprocessing.Barrier` for `multi_process_runner.run`.\n\n  `tf.__internal__.distribute.multi_process_runner.get_barrier()` returns\n  a `multiprocessing.Barrier` object which can be used within `fn` of\n  `tf.__internal__.distribute.multi_process_runner` to wait with\n  `barrier.wait()` call until all other tasks have also reached the\n  `barrier.wait()` call, before they can proceed individually.\n\n  Note that all tasks (subprocesses) have to reach `barrier.wait()` call to\n  proceed. Currently it is not supported to block on only a subset of tasks\n  in the cluster.\n\n  Example:\n  ```python\n\n  def fn():\n    some_work_to_be_done_by_all_tasks()\n\n    tf.__internal__.distribute.multi_process_runner.get_barrier().wait()\n\n    # The barrier guarantees that at this point, all tasks have finished\n    # `some_work_to_be_done_by_all_tasks()`\n    some_other_work_to_be_done_by_all_tasks()\n\n  result = tf.__internal__.distribute.multi_process_runner.run(\n      fn=fn,\n      cluster_spec=(\n          tf.__internal__\n          .distribute.multi_process_runner.create_cluster_spec(\n              num_workers=2)))\n  ```\n\n\n  Returns:\n    A `multiprocessing.Barrier` for `multi_process_runner.run`.\n  '
    if _barrier is None:
        raise ValueError('barrier is not defined. It is likely because you are calling get_barrier() in the main process. get_barrier() can only be called in the subprocesses.')
    return _barrier
_manager = None
_manager_lock = threading.Lock()

def manager():
    if False:
        return 10
    'Returns the multiprocessing manager object for concurrency tools.\n\n  The manager object is useful as it controls a server process that holds\n  the python objects that can be shared across processes. This can be used\n  for parent-subprocess communication:\n\n  ```python\n  manager = multi_process_runner.manager()\n  some_event_happening_in_subprocess = manager.Event()\n  mpr = multi_process_runner.MultiProcessRunner(fn, cluster_spec,\n      args=(some_event_happening_in_subprocess,))\n  mpr.start()\n  some_event_happening_in_subprocess.wait()\n  # Do something that only should after some event happens in subprocess.\n  ```\n\n  Note that the user of multi_process_runner should not create additional\n  `multiprocessing.Manager()` objects; doing so can result in segfault in\n  some cases.\n\n  This method should only be called after multi_process_runner.test_main() is\n  called.\n  '
    _check_initialization()
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = multiprocessing.Manager()
        return _manager

@tf_export('__internal__.distribute.multi_process_runner.test_main', v1=[])
def test_main():
    if False:
        return 10
    "Main function to be called within `__main__` of a test file.\n\n  Any test module that uses\n  `tf.__internal__.distribute.multi_process_runner.run()`\n  must call this instead of regular `test.main()` inside\n  `if __name__ == '__main__':` block, or an error will be raised when\n  `tf.__internal__.distribute.multi_process_runner.run()` is used. This method\n  takes\n  care of needed initialization for launching multiple subprocesses.\n\n  Example:\n  ```python\n  class MyTestClass(tf.test.TestCase):\n    def testSomething(self):\n      # Testing code making use of\n      # `tf.__internal__.distribute.multi_process_runner.run()`.\n\n  if __name__ == '__main__':\n    tf.__internal__.distribute.multi_process_runner.test_main()\n  ```\n  "
    old_tear_down_module = getattr(sys.modules['__main__'], 'tearDownModule', None)

    def tear_down_module():
        if False:
            for i in range(10):
                print('nop')
        _shutdown_all_pool_runners()
        if old_tear_down_module is not None:
            old_tear_down_module()
    setattr(sys.modules['__main__'], 'tearDownModule', tear_down_module)
    multi_process_lib.test_main()