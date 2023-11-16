"""Class MirroredStrategy implementing tf.distribute.Strategy."""
import contextlib
import threading
import weakref
from tensorflow.python import pywrap_tfe
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import shared_variable_creator
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.util import traceback_utils

def _is_gpu_device(device):
    if False:
        for i in range(10):
            print('nop')
    return tf_device.DeviceSpec.from_string(device).device_type == 'GPU'

def call_for_each_replica(strategy, fn, args=None, kwargs=None):
    if False:
        for i in range(10):
            print('nop')
    "Call `fn` on each worker devices(replica).\n\n  It's highly recommended to wrap the call to this function inside a\n  `tf.function`, otherwise the performance is poor.\n\n  Args:\n    strategy: `tf.distribute.Strategy`.\n    fn: function to call on each worker devices.\n    args: positional arguments to `fn`.\n    kwargs: keyword arguments to `fn`.\n\n  Returns:\n    Wrapped returned value of `fn` from all replicas.\n  "
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if isinstance(fn, def_function.Function):
        if fn._jit_compile and all([_is_gpu_device(d) for d in strategy.extended.worker_devices]):
            return _call_for_each_replica(strategy, fn, args, kwargs)
        if strategy not in _cfer_fn_cache:
            _cfer_fn_cache[strategy] = weakref.WeakKeyDictionary()
        wrapped = _cfer_fn_cache[strategy].get(fn)
        if wrapped is None:

            def wrapped_fn(*args, **kwargs):
                if False:
                    print('Hello World!')
                return call_for_each_replica(strategy, fn.python_function, args, kwargs)
            wrapped = fn._clone(python_function=wrapped_fn)
            _cfer_fn_cache[strategy][fn] = wrapped
        return wrapped(*args, **kwargs)
    if context.executing_eagerly():
        logging.log_first_n(logging.WARN, 'Using %s eagerly has significant overhead currently. We will be working on improving this in the future, but for now please wrap `call_for_each_replica` or `experimental_run` or `run` inside a tf.function to get the best performance.' % strategy.__class__.__name__, 5)
    else:
        fn = autograph.tf_convert(fn, autograph_ctx.control_status_ctx())
    return _call_for_each_replica(strategy, fn, args, kwargs)
_cfer_fn_cache = weakref.WeakKeyDictionary()

@contextlib.contextmanager
def _enter_graph(g, eager, creator_stack=None):
    if False:
        i = 10
        return i + 15
    'Context manager for selecting a graph and maybe eager mode.'
    if eager:
        with g.as_default(), context.eager_mode():
            if creator_stack is not None:
                g._variable_creator_stack = creator_stack
            yield
    else:
        with g.as_default():
            if creator_stack is not None:
                g._variable_creator_stack = creator_stack
            yield

@contextlib.contextmanager
def _maybe_enter_eager_mode(eager):
    if False:
        return 10
    if eager:
        with context.eager_mode():
            yield
    else:
        yield

def _cpu_device(device):
    if False:
        i = 10
        return i + 15
    cpu_device = tf_device.DeviceSpec.from_string(device)
    cpu_device = cpu_device.replace(device_type='CPU', device_index=0)
    return cpu_device.to_string()

class _RequestedStop(Exception):
    pass

def _get_thread_local_configuration_callable():
    if False:
        return 10
    if traceback_utils.is_traceback_filtering_enabled():
        thread_local_callables = {traceback_utils.enable_traceback_filtering}
    else:
        thread_local_callables = {traceback_utils.disable_traceback_filtering}
    return thread_local_callables

def _call_for_each_replica(distribution, fn, args, kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Run `fn` in separate threads, once per replica/worker device.\n\n  Args:\n    distribution: the DistributionStrategy object.\n    fn: function to run (will be run once per replica, each in its own thread).\n    args: positional arguments for `fn`\n    kwargs: keyword arguments for `fn`.\n\n  Returns:\n    Merged return value of `fn` across all replicas.\n\n  Raises:\n    RuntimeError: If fn() calls get_replica_context().merge_call() a different\n        number of times from the available devices.\n  '
    run_concurrently = False
    if not context.executing_eagerly():
        ops.get_default_graph().switch_to_thread_local()
    coord = coordinator.Coordinator(clean_stop_exception_types=(_RequestedStop,))
    shared_variable_store = {}
    devices = distribution.extended.worker_devices
    thread_local_callables = _get_thread_local_configuration_callable()
    threads = []
    for index in range(len(devices)):
        variable_creator_fn = shared_variable_creator.make_fn(shared_variable_store, index)
        t = _MirroredReplicaThread(distribution, coord, index, devices, variable_creator_fn, fn, distribute_utils.caching_scope_local, distribute_utils.select_replica(index, args), distribute_utils.select_replica(index, kwargs), thread_local_callables)
        threads.append(t)
    for t in threads:
        t.start()
    try:
        with coord.stop_on_exception():
            all_done = False
            while not all_done and (not coord.should_stop()):
                done = []
                if run_concurrently:
                    for t in threads:
                        t.should_run.set()
                    for t in threads:
                        t.has_paused.wait()
                        t.has_paused.clear()
                        if coord.should_stop():
                            return None
                        done.append(t.done)
                else:
                    for t in threads:
                        t.should_run.set()
                        t.has_paused.wait()
                        t.has_paused.clear()
                        if coord.should_stop():
                            return None
                        done.append(t.done)
                if coord.should_stop():
                    return None
                all_done = all(done)
                if not all_done:
                    if any(done):
                        raise RuntimeError('Some replicas made a different number of replica_context().merge_call() calls.')
                    merge_args = distribute_utils.regroup(tuple((t.merge_args for t in threads)))
                    merge_kwargs = distribute_utils.regroup(tuple((t.merge_kwargs for t in threads)))
                    mtt_captured_name_scope = threads[0].captured_name_scope
                    mtt_captured_var_scope = threads[0].captured_var_scope
                    mtt_captured_control_deps = set()
                    for t in threads:
                        mtt_captured_control_deps.update(t.captured_control_deps)
                    with ops.name_scope(mtt_captured_name_scope), ops.control_dependencies(mtt_captured_control_deps), variable_scope.variable_scope(mtt_captured_var_scope), _maybe_enter_eager_mode(threads[0].merge_call_entered_in_eager):
                        merge_result = threads[0].merge_fn(distribution, *merge_args, **merge_kwargs)
                    for (r, t) in enumerate(threads):
                        t.merge_result = distribute_utils.select_replica(r, merge_result)
    finally:
        for t in threads:
            t.should_run.set()
        coord.join(threads)
    return distribute_utils.regroup(tuple((t.main_result for t in threads)))

class _MirroredReplicaThread(threading.Thread):
    """A thread that runs() a function on a device."""

    def __init__(self, dist, coord, replica_id, devices, variable_creator_fn, fn, caching_scope, args, kwargs, thread_local_callables=None):
        if False:
            print('Hello World!')
        super(_MirroredReplicaThread, self).__init__()
        self.coord = coord
        self.distribution = dist
        self.devices = devices
        self.replica_id = replica_id
        self.replica_id_in_sync_group = dist.extended._get_replica_id_in_sync_group(replica_id)
        self.variable_creator_fn = variable_creator_fn
        self.main_fn = fn
        self.main_args = args
        self.main_kwargs = kwargs
        self.main_result = None
        self.done = False
        self.merge_fn = None
        self.merge_args = None
        self.merge_kwargs = None
        self.merge_result = None
        self.captured_name_scope = None
        self.captured_var_scope = None
        try:
            self.caching_scope_entered = caching_scope.new_cache_scope_count
            self.caching_scope_exited = caching_scope.cache_scope_exited_count
        except AttributeError:
            self.caching_scope_entered = None
            self.caching_scope_exited = None
        self.should_run = threading.Event()
        self.has_paused = threading.Event()
        context.ensure_initialized()
        ctx = context.context()
        self.in_eager = ctx.executing_eagerly()
        self.record_thread_local_summary_state()
        self.record_thread_local_eager_context_state()
        self.context_device_policy = pywrap_tfe.TFE_ContextGetDevicePlacementPolicy(ctx._context_handle)
        self.graph = ops.get_default_graph()
        with ops.init_scope():
            self._init_in_eager = context.executing_eagerly()
            self._init_graph = ops.get_default_graph()
        self._variable_creator_stack = self.graph._variable_creator_stack[:]
        self._var_scope = variable_scope.get_variable_scope()
        self._name_scope = self.graph.get_name_scope()
        if self._name_scope:
            self._name_scope += '/'
        if self.replica_id > 0:
            if not self._name_scope:
                self._name_scope = ''
            self._name_scope += 'replica_%d/' % self.replica_id
        self._thread_local_callables = thread_local_callables

    def run(self):
        if False:
            return 10
        self.should_run.wait()
        self.should_run.clear()
        try:
            if self.coord.should_stop():
                return
            self.restore_thread_local_summary_state()
            self.restore_thread_local_callable()
            self.restore_thread_local_eager_context_state()
            if self.caching_scope_entered is not None and self.caching_scope_exited is not None:
                distribute_utils.caching_scope_local.new_cache_scope_count = self.caching_scope_entered
                distribute_utils.caching_scope_local.cache_scope_exited_count = self.caching_scope_exited
            with self.coord.stop_on_exception(), _enter_graph(self._init_graph, self._init_in_eager), _enter_graph(self.graph, self.in_eager, self._variable_creator_stack), context.device_policy(self.context_device_policy), _MirroredReplicaContext(self.distribution, self.replica_id_in_sync_group), ops.device(self.devices[self.replica_id]), ops.name_scope(self._name_scope), variable_scope.variable_scope(self._var_scope, reuse=self.replica_id > 0), variable_scope.variable_creator_scope(self.variable_creator_fn):
                self.main_result = self.main_fn(*self.main_args, **self.main_kwargs)
                self.done = True
        finally:
            self.has_paused.set()

    def record_thread_local_summary_state(self):
        if False:
            return 10
        'Record the thread local summary state in self.'
        summary_state = summary_ops_v2._summary_state
        self._summary_step = summary_state.step
        self._summary_writer = summary_state.writer
        self._summary_recording = summary_state.is_recording
        self._summary_recording_distribution_strategy = summary_state.is_recording_distribution_strategy

    def restore_thread_local_summary_state(self):
        if False:
            i = 10
            return i + 15
        'Restore thread local summary state from self.'
        summary_state = summary_ops_v2._summary_state
        summary_state.step = self._summary_step
        summary_state.writer = self._summary_writer
        summary_state.is_recording = self._summary_recording
        summary_state.is_recording_distribution_strategy = self._summary_recording_distribution_strategy

    def record_thread_local_eager_context_state(self):
        if False:
            print('Hello World!')
        ctx = context.context()
        eager_context_state = ctx._thread_local_data
        self._eager_context_op_callbacks = eager_context_state.op_callbacks

    def restore_thread_local_eager_context_state(self):
        if False:
            return 10
        ctx = context.context()
        eager_context_state = ctx._thread_local_data
        eager_context_state.op_callbacks = self._eager_context_op_callbacks

    def restore_thread_local_callable(self):
        if False:
            i = 10
            return i + 15
        if self._thread_local_callables:
            for fn in self._thread_local_callables:
                fn()

class _MirroredReplicaContext(distribute_lib.ReplicaContext):
    """ReplicaContext for synchronized replica."""

    def _merge_call(self, fn, args, kwargs):
        if False:
            return 10
        '`merge_call()` implementation for synchronized replica.\n\n    This pauses the current replica thread and passes `fn` and its arguments to\n    the main thread. The main thread will wait until all replicas pause, then\n    invoke `fn` with grouped arguments. The current replica thread will continue\n    after `fn` completes.\n\n    See `_call_for_each_replica` for the logic in the main thread.\n\n    Args:\n      fn: a function that is called in cross replica context with grouped\n        arguments from each replica. `fn` should returns grouped values.\n      args: positional arguments to `fn`.\n      kwargs: keyward arguments to `fn`.\n\n    Returns:\n      Return value of `fn` for the current replica.\n\n    Raises:\n      RuntimeError: when merge_call happens in a different graph, e.g. in a\n        different tf.function, which is not supported now.\n      _RequestedStop: when stop is requested.\n\n    '
        t = threading.current_thread()
        assert isinstance(t, _MirroredReplicaThread)
        t.merge_fn = fn
        t.merge_args = args
        t.merge_kwargs = kwargs
        t.captured_name_scope = t.graph.get_name_scope()
        if t.captured_name_scope:
            t.captured_name_scope += '/'
        t.captured_var_scope = variable_scope.get_variable_scope()
        t.captured_control_deps = t.graph._current_control_dependencies()
        t.merge_call_entered_in_eager = context.context().executing_eagerly()
        if ops.get_default_graph() != t.graph:
            raise RuntimeError('`merge_call` called while defining a new graph or a tf.function. This can often happen if the function `fn` passed to `strategy.run()` contains a nested `@tf.function`, and the nested `@tf.function` contains a synchronization point, such as aggregating gradients (e.g, optimizer.apply_gradients), or if the function `fn` uses a control flow statement which contains a synchronization point in the body. Such behaviors are not yet supported. Instead, please avoid nested `tf.function`s or control flow statements that may potentially cross a synchronization boundary, for example, wrap the `fn` passed to `strategy.run` or the entire `strategy.run` inside a `tf.function` or move the control flow out of `fn`. If you are subclassing a `tf.keras.Model`, please avoid decorating overridden methods `test_step` and `train_step` in `tf.function`.')
        t.has_paused.set()
        t.should_run.wait()
        t.should_run.clear()
        if t.coord.should_stop():
            raise _RequestedStop()
        t.merge_call_entered_in_eager = None
        return t.merge_result

    @property
    def devices(self):
        if False:
            for i in range(10):
                print('nop')
        distribute_lib.require_replica_context(self)
        return [self._strategy.extended.worker_devices_by_replica[self._replica_id_in_sync_group]]