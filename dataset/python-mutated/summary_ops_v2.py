"""Operations to emit summaries."""
import abc
import collections
import functools
import os
import re
import threading
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.dtensor.python import api as dtensor_api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import profiler as _profiler
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_summary_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import summary_op_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import resource
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
_SUMMARY_WRITER_INIT_COLLECTION_NAME = '_SUMMARY_WRITER_V2'

class _SummaryState(threading.local):

    def __init__(self):
        if False:
            return 10
        super(_SummaryState, self).__init__()
        self.is_recording = None
        self.is_recording_distribution_strategy = True
        self.writer = None
        self.step = None
_summary_state = _SummaryState()

class _SummaryContextManager:
    """Context manager to implement SummaryWriter.as_default()."""

    def __init__(self, writer, step=None):
        if False:
            for i in range(10):
                print('nop')
        self._writer = writer
        self._step = step
        self._old_writer = None
        self._old_step = None

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self._old_writer = _summary_state.writer
        _summary_state.writer = self._writer
        if self._step is not None:
            self._old_step = _summary_state.step
            _summary_state.step = self._step
        return self._writer

    def __exit__(self, *exc):
        if False:
            i = 10
            return i + 15
        _summary_state.writer.flush()
        _summary_state.writer = self._old_writer
        if self._step is not None:
            _summary_state.step = self._old_step
        return False

def _should_record_summaries_internal(default_state):
    if False:
        while True:
            i = 10
    'Returns boolean Tensor if summaries should/shouldn\'t be recorded.\n\n  Now the summary condition is decided by logical "and" of below conditions:\n  First, summary writer must be set. Given this constraint is met,\n  ctx.summary_recording and ctx.summary_recording_distribution_strategy.\n  The former one is usually set by user, and the latter one is controlled\n  by DistributionStrategy (tf.distribute.ReplicaContext).\n\n  Args:\n    default_state: can be True or False. The default summary behavior when\n    summary writer is set and the user does not specify\n    ctx.summary_recording and ctx.summary_recording_distribution_strategy\n    is True.\n  '
    if _summary_state.writer is None:
        return constant_op.constant(False)
    if not callable(_summary_state.is_recording):
        static_cond = tensor_util.constant_value(_summary_state.is_recording)
        if static_cond is not None and (not static_cond):
            return constant_op.constant(False)
    resolve = lambda x: x() if callable(x) else x
    cond_distributed = resolve(_summary_state.is_recording_distribution_strategy)
    cond = resolve(_summary_state.is_recording)
    if cond is None:
        cond = default_state
    return math_ops.logical_and(cond_distributed, cond)

@tf_export('summary.should_record_summaries', v1=[])
def should_record_summaries():
    if False:
        i = 10
        return i + 15
    'Returns boolean Tensor which is True if summaries will be recorded.\n\n  If no default summary writer is currently registered, this always returns\n  False. Otherwise, this reflects the recording condition has been set via\n  `tf.summary.record_if()` (except that it may return False for some replicas\n  when using `tf.distribute.Strategy`). If no recording condition is active,\n  it defaults to True.\n  '
    return _should_record_summaries_internal(default_state=True)

def _legacy_contrib_should_record_summaries():
    if False:
        return 10
    'Returns boolean Tensor which is true if summaries should be recorded.'
    return _should_record_summaries_internal(default_state=False)

@tf_export('summary.record_if', v1=[])
@tf_contextlib.contextmanager
def record_if(condition):
    if False:
        for i in range(10):
            print('nop')
    'Sets summary recording on or off per the provided boolean value.\n\n  The provided value can be a python boolean, a scalar boolean Tensor, or\n  or a callable providing such a value; if a callable is passed it will be\n  invoked on-demand to determine whether summary writing will occur.  Note that\n  when calling record_if() in an eager mode context, if you intend to provide a\n  varying condition like `step % 100 == 0`, you must wrap this in a\n  callable to avoid immediate eager evaluation of the condition.  In particular,\n  using a callable is the only way to have your condition evaluated as part of\n  the traced body of an @tf.function that is invoked from within the\n  `record_if()` context.\n\n  Args:\n    condition: can be True, False, a bool Tensor, or a callable providing such.\n\n  Yields:\n    Returns a context manager that sets this value on enter and restores the\n    previous value on exit.\n  '
    old = _summary_state.is_recording
    try:
        _summary_state.is_recording = condition
        yield
    finally:
        _summary_state.is_recording = old

def has_default_writer():
    if False:
        i = 10
        return i + 15
    'Returns a boolean indicating whether a default summary writer exists.'
    return _summary_state.writer is not None

def record_summaries_every_n_global_steps(n, global_step=None):
    if False:
        while True:
            i = 10
    'Sets the should_record_summaries Tensor to true if global_step % n == 0.'
    if global_step is None:
        global_step = training_util.get_or_create_global_step()
    with ops.device('cpu:0'):
        should = lambda : math_ops.equal(global_step % n, 0)
        if not context.executing_eagerly():
            should = should()
    return record_if(should)

def always_record_summaries():
    if False:
        i = 10
        return i + 15
    'Sets the should_record_summaries Tensor to always true.'
    return record_if(True)

def never_record_summaries():
    if False:
        i = 10
        return i + 15
    'Sets the should_record_summaries Tensor to always false.'
    return record_if(False)

@tf_export('summary.experimental.get_step', v1=[])
def get_step():
    if False:
        i = 10
        return i + 15
    'Returns the default summary step for the current thread.\n\n  Returns:\n    The step set by `tf.summary.experimental.set_step()` if one has been set,\n    otherwise None.\n  '
    return _summary_state.step

@tf_export('summary.experimental.set_step', v1=[])
def set_step(step):
    if False:
        print('Hello World!')
    'Sets the default summary step for the current thread.\n\n  For convenience, this function sets a default value for the `step` parameter\n  used in summary-writing functions elsewhere in the API so that it need not\n  be explicitly passed in every such invocation. The value can be a constant\n  or a variable, and can be retrieved via `tf.summary.experimental.get_step()`.\n\n  Note: when using this with @tf.functions, the step value will be captured at\n  the time the function is traced, so changes to the step outside the function\n  will not be reflected inside the function unless using a `tf.Variable` step.\n\n  Args:\n    step: An `int64`-castable default step value, or None to unset.\n  '
    _summary_state.step = step

@tf_export('summary.SummaryWriter', v1=[])
class SummaryWriter(metaclass=abc.ABCMeta):
    """Interface representing a stateful summary writer object."""

    def set_as_default(self, step=None):
        if False:
            for i in range(10):
                print('nop')
        'Enables this summary writer for the current thread.\n\n    For convenience, if `step` is not None, this function also sets a default\n    value for the `step` parameter used in summary-writing functions elsewhere\n    in the API so that it need not be explicitly passed in every such\n    invocation. The value can be a constant or a variable.\n\n    Note: when setting `step` in a @tf.function, the step value will be\n    captured at the time the function is traced, so changes to the step outside\n    the function will not be reflected inside the function unless using\n    a `tf.Variable` step.\n\n    Args:\n      step: An `int64`-castable default step value, or `None`. When not `None`,\n        the current step is modified to the given value. When `None`, the\n        current step is not modified.\n    '
        self.as_default(step).__enter__()

    def as_default(self, step=None):
        if False:
            while True:
                i = 10
        'Returns a context manager that enables summary writing.\n\n    For convenience, if `step` is not None, this function also sets a default\n    value for the `step` parameter used in summary-writing functions elsewhere\n    in the API so that it need not be explicitly passed in every such\n    invocation. The value can be a constant or a variable.\n\n    Note: when setting `step` in a @tf.function, the step value will be\n    captured at the time the function is traced, so changes to the step outside\n    the function will not be reflected inside the function unless using\n    a `tf.Variable` step.\n\n    For example, `step` can be used as:\n\n    ```python\n    with writer_a.as_default(step=10):\n      tf.summary.scalar(tag, value)   # Logged to writer_a with step 10\n      with writer_b.as_default(step=20):\n        tf.summary.scalar(tag, value) # Logged to writer_b with step 20\n      tf.summary.scalar(tag, value)   # Logged to writer_a with step 10\n    ```\n\n    Args:\n      step: An `int64`-castable default step value, or `None`. When not `None`,\n        the current step is captured, replaced by a given one, and the original\n        one is restored when the context manager exits. When `None`, the current\n        step is not modified (and not restored when the context manager exits).\n\n    Returns:\n      The context manager.\n    '
        return _SummaryContextManager(self, step)

    def init(self):
        if False:
            i = 10
            return i + 15
        'Initializes the summary writer.'
        raise NotImplementedError()

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        'Flushes any buffered data.'
        raise NotImplementedError()

    def close(self):
        if False:
            while True:
                i = 10
        'Flushes and closes the summary writer.'
        raise NotImplementedError()

class _ResourceSummaryWriter(SummaryWriter):
    """Implementation of SummaryWriter using a SummaryWriterInterface resource."""

    def __init__(self, create_fn, init_op_fn, mesh=None):
        if False:
            for i in range(10):
                print('nop')
        if mesh is not None:
            with dtensor_api.default_mesh(mesh.host_mesh()):
                self._resource = create_fn()
                self._init_op = init_op_fn(self._resource)
        else:
            self._resource = create_fn()
            self._init_op = init_op_fn(self._resource)
        self._closed = False
        if context.executing_eagerly():
            self._set_up_resource_deleter()
        else:
            ops.add_to_collection(_SUMMARY_WRITER_INIT_COLLECTION_NAME, self._init_op)
        self._mesh = mesh

    def _set_up_resource_deleter(self):
        if False:
            for i in range(10):
                print('nop')
        self._resource_deleter = resource_variable_ops.EagerResourceDeleter(handle=self._resource, handle_device='cpu:0')

    def set_as_default(self, step=None):
        if False:
            for i in range(10):
                print('nop')
        'See `SummaryWriter.set_as_default`.'
        if context.executing_eagerly() and self._closed:
            raise RuntimeError(f'SummaryWriter {self!r} is already closed')
        super().set_as_default(step)

    def as_default(self, step=None):
        if False:
            print('Hello World!')
        'See `SummaryWriter.as_default`.'
        if context.executing_eagerly() and self._closed:
            raise RuntimeError(f'SummaryWriter {self!r} is already closed')
        return super().as_default(step)

    def init(self):
        if False:
            i = 10
            return i + 15
        'See `SummaryWriter.init`.'
        if context.executing_eagerly() and self._closed:
            raise RuntimeError(f'SummaryWriter {self!r} is already closed')
        return self._init_op

    def flush(self):
        if False:
            return 10
        'See `SummaryWriter.flush`.'
        if context.executing_eagerly() and self._closed:
            return
        with ops.device('cpu:0'):
            return gen_summary_ops.flush_summary_writer(self._resource)

    def close(self):
        if False:
            i = 10
            return i + 15
        'See `SummaryWriter.close`.'
        if context.executing_eagerly() and self._closed:
            return
        try:
            with ops.control_dependencies([self.flush()]):
                with ops.device('cpu:0'):
                    return gen_summary_ops.close_summary_writer(self._resource)
        finally:
            if context.executing_eagerly():
                self._closed = True

class _MultiMetaclass(type(_ResourceSummaryWriter), type(resource.TrackableResource)):
    pass

class _TrackableResourceSummaryWriter(_ResourceSummaryWriter, resource.TrackableResource, metaclass=_MultiMetaclass):
    """A `_ResourceSummaryWriter` subclass that implements `TrackableResource`."""

    def __init__(self, create_fn, init_op_fn, mesh=None):
        if False:
            i = 10
            return i + 15
        resource.TrackableResource.__init__(self, device='/CPU:0')
        self._create_fn = create_fn
        self._init_op_fn = init_op_fn
        _ResourceSummaryWriter.__init__(self, create_fn=lambda : self.resource_handle, init_op_fn=init_op_fn, mesh=mesh)

    def _create_resource(self):
        if False:
            i = 10
            return i + 15
        return self._create_fn()

    def _initialize(self):
        if False:
            for i in range(10):
                print('nop')
        return self._init_op_fn(self.resource_handle)

    def _destroy_resource(self):
        if False:
            for i in range(10):
                print('nop')
        gen_resource_variable_ops.destroy_resource_op(self.resource_handle, ignore_lookup_error=True)

    def _set_up_resource_deleter(self):
        if False:
            print('Hello World!')
        pass

class _LegacyResourceSummaryWriter(SummaryWriter):
    """Legacy resource-backed SummaryWriter for tf.contrib.summary."""

    def __init__(self, resource, init_op_fn):
        if False:
            i = 10
            return i + 15
        self._resource = resource
        self._init_op_fn = init_op_fn
        init_op = self.init()
        if context.executing_eagerly():
            self._resource_deleter = resource_variable_ops.EagerResourceDeleter(handle=self._resource, handle_device='cpu:0')
        else:
            ops.add_to_collection(_SUMMARY_WRITER_INIT_COLLECTION_NAME, init_op)

    def init(self):
        if False:
            return 10
        'See `SummaryWriter.init`.'
        return self._init_op_fn(self._resource)

    def flush(self):
        if False:
            return 10
        'See `SummaryWriter.flush`.'
        with ops.device('cpu:0'):
            return gen_summary_ops.flush_summary_writer(self._resource)

    def close(self):
        if False:
            while True:
                i = 10
        'See `SummaryWriter.close`.'
        with ops.control_dependencies([self.flush()]):
            with ops.device('cpu:0'):
                return gen_summary_ops.close_summary_writer(self._resource)

class _NoopSummaryWriter(SummaryWriter):
    """A summary writer that does nothing, for create_noop_writer()."""

    def set_as_default(self, step=None):
        if False:
            while True:
                i = 10
        pass

    @tf_contextlib.contextmanager
    def as_default(self, step=None):
        if False:
            while True:
                i = 10
        yield

    def init(self):
        if False:
            i = 10
            return i + 15
        pass

    def flush(self):
        if False:
            print('Hello World!')
        pass

    def close(self):
        if False:
            return 10
        pass

@tf_export(v1=['summary.initialize'])
def initialize(graph=None, session=None):
    if False:
        return 10
    "Initializes summary writing for graph execution mode.\n\n  This operation is a no-op when executing eagerly.\n\n  This helper method provides a higher-level alternative to using\n  `tf.contrib.summary.summary_writer_initializer_op` and\n  `tf.contrib.summary.graph`.\n\n  Most users will also want to call `tf.compat.v1.train.create_global_step`\n  which can happen before or after this function is called.\n\n  Args:\n    graph: A `tf.Graph` or `tf.compat.v1.GraphDef` to output to the writer.\n      This function will not write the default graph by default. When\n      writing to an event log file, the associated step will be zero.\n    session: So this method can call `tf.Session.run`. This defaults\n      to `tf.compat.v1.get_default_session`.\n\n  Raises:\n    RuntimeError: If  the current thread has no default\n      `tf.contrib.summary.SummaryWriter`.\n    ValueError: If session wasn't passed and no default session.\n  "
    if context.executing_eagerly():
        return
    if _summary_state.writer is None:
        raise RuntimeError('No default tf.contrib.summary.SummaryWriter found')
    if session is None:
        session = ops.get_default_session()
        if session is None:
            raise ValueError('Argument `session must be passed if no default session exists')
    session.run(summary_writer_initializer_op())
    if graph is not None:
        data = _serialize_graph(graph)
        x = array_ops.placeholder(dtypes.string)
        session.run(graph_v1(x, 0), feed_dict={x: data})

@tf_export('summary.create_file_writer', v1=[])
def create_file_writer_v2(logdir, max_queue=None, flush_millis=None, filename_suffix=None, name=None, experimental_trackable=False, experimental_mesh=None):
    if False:
        return 10
    'Creates a summary file writer for the given log directory.\n\n  Args:\n    logdir: a string specifying the directory in which to write an event file.\n    max_queue: the largest number of summaries to keep in a queue; will flush\n      once the queue gets bigger than this. Defaults to 10.\n    flush_millis: the largest interval between flushes. Defaults to 120,000.\n    filename_suffix: optional suffix for the event file name. Defaults to `.v2`.\n    name: a name for the op that creates the writer.\n    experimental_trackable: a boolean that controls whether the returned writer\n      will be a `TrackableResource`, which makes it compatible with SavedModel\n      when used as a `tf.Module` property.\n    experimental_mesh: a `tf.experimental.dtensor.Mesh` instance. When running\n      with DTensor, the mesh (experimental_mesh.host_mesh()) will be used for\n      bringing all the DTensor logging from accelerator to CPU mesh.\n\n  Returns:\n    A SummaryWriter object.\n  '
    if logdir is None:
        raise ValueError('Argument `logdir` cannot be None')
    inside_function = ops.inside_function()
    with ops.name_scope(name, 'create_file_writer') as scope, ops.device('cpu:0'):
        with ops.init_scope():
            if context.executing_eagerly():
                _check_create_file_writer_args(inside_function, logdir=logdir, max_queue=max_queue, flush_millis=flush_millis, filename_suffix=filename_suffix)
            logdir = ops.convert_to_tensor(logdir, dtype=dtypes.string)
            if max_queue is None:
                max_queue = constant_op.constant(10)
            if flush_millis is None:
                flush_millis = constant_op.constant(2 * 60 * 1000)
            if filename_suffix is None:
                filename_suffix = constant_op.constant('.v2')

            def create_fn():
                if False:
                    while True:
                        i = 10
                if context.executing_eagerly():
                    shared_name = context.anonymous_name()
                else:
                    shared_name = ops.name_from_scope_name(scope)
                return gen_summary_ops.summary_writer(shared_name=shared_name, name=name)
            init_op_fn = functools.partial(gen_summary_ops.create_summary_file_writer, logdir=logdir, max_queue=max_queue, flush_millis=flush_millis, filename_suffix=filename_suffix)
            if experimental_trackable:
                return _TrackableResourceSummaryWriter(create_fn=create_fn, init_op_fn=init_op_fn, mesh=experimental_mesh)
            else:
                return _ResourceSummaryWriter(create_fn=create_fn, init_op_fn=init_op_fn, mesh=experimental_mesh)

def create_file_writer(logdir, max_queue=None, flush_millis=None, filename_suffix=None, name=None):
    if False:
        print('Hello World!')
    'Creates a summary file writer in the current context under the given name.\n\n  Args:\n    logdir: a string, or None. If a string, creates a summary file writer\n     which writes to the directory named by the string. If None, returns\n     a mock object which acts like a summary writer but does nothing,\n     useful to use as a context manager.\n    max_queue: the largest number of summaries to keep in a queue; will\n     flush once the queue gets bigger than this. Defaults to 10.\n    flush_millis: the largest interval between flushes. Defaults to 120,000.\n    filename_suffix: optional suffix for the event file name. Defaults to `.v2`.\n    name: Shared name for this SummaryWriter resource stored to default\n      Graph. Defaults to the provided logdir prefixed with `logdir:`. Note: if a\n      summary writer resource with this shared name already exists, the returned\n      SummaryWriter wraps that resource and the other arguments have no effect.\n\n  Returns:\n    Either a summary writer or an empty object which can be used as a\n    summary writer.\n  '
    if logdir is None:
        return _NoopSummaryWriter()
    logdir = str(logdir)
    with ops.device('cpu:0'):
        if max_queue is None:
            max_queue = constant_op.constant(10)
        if flush_millis is None:
            flush_millis = constant_op.constant(2 * 60 * 1000)
        if filename_suffix is None:
            filename_suffix = constant_op.constant('.v2')
        if name is None:
            name = 'logdir:' + logdir
        resource = gen_summary_ops.summary_writer(shared_name=name)
        return _LegacyResourceSummaryWriter(resource=resource, init_op_fn=functools.partial(gen_summary_ops.create_summary_file_writer, logdir=logdir, max_queue=max_queue, flush_millis=flush_millis, filename_suffix=filename_suffix))

@tf_export('summary.create_noop_writer', v1=[])
def create_noop_writer():
    if False:
        return 10
    'Returns a summary writer that does nothing.\n\n  This is useful as a placeholder in code that expects a context manager.\n  '
    return _NoopSummaryWriter()

def _cleanse_string(name, pattern, value):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(value, str) and pattern.search(value) is None:
        raise ValueError(f'{name} ({value}) must match {pattern.pattern}')
    return ops.convert_to_tensor(value, dtypes.string)

def _nothing():
    if False:
        print('Hello World!')
    'Convenient else branch for when summaries do not record.'
    return constant_op.constant(False)

@tf_export(v1=['summary.all_v2_summary_ops'])
def all_v2_summary_ops():
    if False:
        while True:
            i = 10
    'Returns all V2-style summary ops defined in the current default graph.\n\n  This includes ops from TF 2.0 tf.summary and TF 1.x tf.contrib.summary (except\n  for `tf.contrib.summary.graph` and `tf.contrib.summary.import_event`), but\n  does *not* include TF 1.x tf.summary ops.\n\n  Returns:\n    List of summary ops, or None if called under eager execution.\n  '
    if context.executing_eagerly():
        return None
    return ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)

def summary_writer_initializer_op():
    if False:
        while True:
            i = 10
    'Graph-mode only. Returns the list of ops to create all summary writers.\n\n  Returns:\n    The initializer ops.\n\n  Raises:\n    RuntimeError: If in Eager mode.\n  '
    if context.executing_eagerly():
        raise RuntimeError('tf.contrib.summary.summary_writer_initializer_op is only supported in graph mode.')
    return ops.get_collection(_SUMMARY_WRITER_INIT_COLLECTION_NAME)
_INVALID_SCOPE_CHARACTERS = re.compile('[^-_/.A-Za-z0-9]')

@tf_export('summary.experimental.summary_scope', v1=[])
@tf_contextlib.contextmanager
def summary_scope(name, default_name='summary', values=None):
    if False:
        i = 10
        return i + 15
    'Experimental context manager for use when defining a custom summary op.\n\n  This behaves similarly to `tf.name_scope`, except that it returns a generated\n  summary tag in addition to the scope name. The tag is structurally similar to\n  the scope name - derived from the user-provided name, prefixed with enclosing\n  name scopes if any - but we relax the constraint that it be uniquified, as\n  well as the character set limitation (so the user-provided name can contain\n  characters not legal for scope names; in the scope name these are removed).\n\n  This makes the summary tag more predictable and consistent for the user.\n\n  For example, to define a new summary op called `my_op`:\n\n  ```python\n  def my_op(name, my_value, step):\n    with tf.summary.summary_scope(name, "MyOp", [my_value]) as (tag, scope):\n      my_value = tf.convert_to_tensor(my_value)\n      return tf.summary.write(tag, my_value, step=step)\n  ```\n\n  Args:\n    name: string name for the summary.\n    default_name: Optional; if provided, used as default name of the summary.\n    values: Optional; passed as `values` parameter to name_scope.\n\n  Yields:\n    A tuple `(tag, scope)` as described above.\n  '
    name = name or default_name
    current_scope = ops.get_name_scope()
    tag = current_scope + '/' + name if current_scope else name
    name = _INVALID_SCOPE_CHARACTERS.sub('', name) or None
    with ops.name_scope(name, default_name, values, skip_on_eager=False) as scope:
        yield (tag, scope)

@tf_export('summary.write', v1=[])
def write(tag, tensor, step=None, metadata=None, name=None):
    if False:
        return 10
    'Writes a generic summary to the default SummaryWriter if one exists.\n\n  This exists primarily to support the definition of type-specific summary ops\n  like scalar() and image(), and is not intended for direct use unless defining\n  a new type-specific summary op.\n\n  Args:\n    tag: string tag used to identify the summary (e.g. in TensorBoard), usually\n      generated with `tf.summary.summary_scope`\n    tensor: the Tensor holding the summary data to write or a callable that\n      returns this Tensor. If a callable is passed, it will only be called when\n      a default SummaryWriter exists and the recording condition specified by\n      `record_if()` is met.\n    step: Explicit `int64`-castable monotonic step value for this summary. If\n      omitted, this defaults to `tf.summary.experimental.get_step()`, which must\n      not be None.\n    metadata: Optional SummaryMetadata, as a proto or serialized bytes\n    name: Optional string name for this op.\n\n  Returns:\n    True on success, or false if no summary was written because no default\n    summary writer was available.\n\n  Raises:\n    ValueError: if a default writer exists, but no step was provided and\n      `tf.summary.experimental.get_step()` is None.\n  '
    with ops.name_scope(name, 'write_summary') as scope:
        if _summary_state.writer is None:
            return constant_op.constant(False)
        if step is None:
            step = get_step()
        if metadata is None:
            serialized_metadata = b''
        elif hasattr(metadata, 'SerializeToString'):
            serialized_metadata = metadata.SerializeToString()
        else:
            serialized_metadata = metadata

        def record():
            if False:
                while True:
                    i = 10
            'Record the actual summary and return True.'
            if step is None:
                raise ValueError('No step set. Please specify one either through the `step` argument or through tf.summary.experimental.set_step()')
            with ops.device('cpu:0'):
                summary_tensor = tensor() if callable(tensor) else array_ops.identity(tensor)
                writer = _summary_state.writer
                summary_value = _maybe_convert_tensor_to_dtensor(writer, summary_tensor)
                step_value = _maybe_convert_tensor_to_dtensor(writer, step)
                write_summary_op = gen_summary_ops.write_summary(writer._resource, step_value, summary_value, tag, serialized_metadata, name=scope)
                with ops.control_dependencies([write_summary_op]):
                    return constant_op.constant(True)
        op = smart_cond.smart_cond(should_record_summaries(), record, _nothing, name='summary_cond')
        if not context.executing_eagerly():
            ops.add_to_collection(ops.GraphKeys._SUMMARY_COLLECTION, op)
        return op

@tf_export('summary.experimental.write_raw_pb', v1=[])
def write_raw_pb(tensor, step=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Writes a summary using raw `tf.compat.v1.Summary` protocol buffers.\n\n  Experimental: this exists to support the usage of V1-style manual summary\n  writing (via the construction of a `tf.compat.v1.Summary` protocol buffer)\n  with the V2 summary writing API.\n\n  Args:\n    tensor: the string Tensor holding one or more serialized `Summary` protobufs\n    step: Explicit `int64`-castable monotonic step value for this summary. If\n      omitted, this defaults to `tf.summary.experimental.get_step()`, which must\n      not be None.\n    name: Optional string name for this op.\n\n  Returns:\n    True on success, or false if no summary was written because no default\n    summary writer was available.\n\n  Raises:\n    ValueError: if a default writer exists, but no step was provided and\n      `tf.summary.experimental.get_step()` is None.\n  '
    with ops.name_scope(name, 'write_raw_pb') as scope:
        if _summary_state.writer is None:
            return constant_op.constant(False)
        if step is None:
            step = get_step()
            if step is None:
                raise ValueError('No step set. Please specify one either through the `step` argument or through tf.summary.experimental.set_step()')

        def record():
            if False:
                i = 10
                return i + 15
            'Record the actual summary and return True.'
            with ops.device('cpu:0'):
                raw_summary_op = gen_summary_ops.write_raw_proto_summary(_summary_state.writer._resource, step, array_ops.identity(tensor), name=scope)
                with ops.control_dependencies([raw_summary_op]):
                    return constant_op.constant(True)
        with ops.device('cpu:0'):
            op = smart_cond.smart_cond(should_record_summaries(), record, _nothing, name='summary_cond')
            if not context.executing_eagerly():
                ops.add_to_collection(ops.GraphKeys._SUMMARY_COLLECTION, op)
            return op

def summary_writer_function(name, tensor, function, family=None):
    if False:
        while True:
            i = 10
    "Helper function to write summaries.\n\n  Args:\n    name: name of the summary\n    tensor: main tensor to form the summary\n    function: function taking a tag and a scope which writes the summary\n    family: optional, the summary's family\n\n  Returns:\n    The result of writing the summary.\n  "
    name_scope = ops.get_name_scope()
    if name_scope:
        name_scope += '/'

    def record():
        if False:
            while True:
                i = 10
        with ops.name_scope(name_scope), summary_op_util.summary_scope(name, family, values=[tensor]) as (tag, scope):
            with ops.control_dependencies([function(tag, scope)]):
                return constant_op.constant(True)
    if _summary_state.writer is None:
        return control_flow_ops.no_op()
    with ops.device('cpu:0'):
        op = smart_cond.smart_cond(_legacy_contrib_should_record_summaries(), record, _nothing, name='')
        if not context.executing_eagerly():
            ops.add_to_collection(ops.GraphKeys._SUMMARY_COLLECTION, op)
    return op

def generic(name, tensor, metadata=None, family=None, step=None):
    if False:
        for i in range(10):
            print('nop')
    'Writes a tensor summary if possible.'

    def function(tag, scope):
        if False:
            print('Hello World!')
        if metadata is None:
            serialized_metadata = constant_op.constant('')
        elif hasattr(metadata, 'SerializeToString'):
            serialized_metadata = constant_op.constant(metadata.SerializeToString())
        else:
            serialized_metadata = metadata
        return gen_summary_ops.write_summary(_summary_state.writer._resource, _choose_step(step), array_ops.identity(tensor), tag, serialized_metadata, name=scope)
    return summary_writer_function(name, tensor, function, family=family)

def scalar(name, tensor, family=None, step=None):
    if False:
        i = 10
        return i + 15
    "Writes a scalar summary if possible.\n\n  Unlike `tf.contrib.summary.generic` this op may change the dtype\n  depending on the writer, for both practical and efficiency concerns.\n\n  Args:\n    name: An arbitrary name for this summary.\n    tensor: A `tf.Tensor` Must be one of the following types:\n      `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`,\n      `int8`, `uint16`, `half`, `uint32`, `uint64`.\n    family: Optional, the summary's family.\n    step: The `int64` monotonic step variable, which defaults\n      to `tf.compat.v1.train.get_global_step`.\n\n  Returns:\n    The created `tf.Operation` or a `tf.no_op` if summary writing has\n    not been enabled for this context.\n  "

    def function(tag, scope):
        if False:
            for i in range(10):
                print('nop')
        return gen_summary_ops.write_scalar_summary(_summary_state.writer._resource, _choose_step(step), tag, array_ops.identity(tensor), name=scope)
    return summary_writer_function(name, tensor, function, family=family)

def histogram(name, tensor, family=None, step=None):
    if False:
        i = 10
        return i + 15
    'Writes a histogram summary if possible.'

    def function(tag, scope):
        if False:
            i = 10
            return i + 15
        return gen_summary_ops.write_histogram_summary(_summary_state.writer._resource, _choose_step(step), tag, array_ops.identity(tensor), name=scope)
    return summary_writer_function(name, tensor, function, family=family)

def image(name, tensor, bad_color=None, max_images=3, family=None, step=None):
    if False:
        print('Hello World!')
    'Writes an image summary if possible.'

    def function(tag, scope):
        if False:
            i = 10
            return i + 15
        bad_color_ = constant_op.constant([255, 0, 0, 255], dtype=dtypes.uint8) if bad_color is None else bad_color
        return gen_summary_ops.write_image_summary(_summary_state.writer._resource, _choose_step(step), tag, array_ops.identity(tensor), bad_color_, max_images, name=scope)
    return summary_writer_function(name, tensor, function, family=family)

def audio(name, tensor, sample_rate, max_outputs, family=None, step=None):
    if False:
        while True:
            i = 10
    'Writes an audio summary if possible.'

    def function(tag, scope):
        if False:
            i = 10
            return i + 15
        return gen_summary_ops.write_audio_summary(_summary_state.writer._resource, _choose_step(step), tag, array_ops.identity(tensor), sample_rate=sample_rate, max_outputs=max_outputs, name=scope)
    return summary_writer_function(name, tensor, function, family=family)

def graph_v1(param, step=None, name=None):
    if False:
        i = 10
        return i + 15
    "Writes a TensorFlow graph to the summary interface.\n\n  The graph summary is, strictly speaking, not a summary. Conditions\n  like `tf.summary.should_record_summaries` do not apply. Only\n  a single graph can be associated with a particular run. If multiple\n  graphs are written, then only the last one will be considered by\n  TensorBoard.\n\n  When not using eager execution mode, the user should consider passing\n  the `graph` parameter to `tf.compat.v1.summary.initialize` instead of\n  calling this function. Otherwise special care needs to be taken when\n  using the graph to record the graph.\n\n  Args:\n    param: A `tf.Tensor` containing a serialized graph proto. When\n      eager execution is enabled, this function will automatically\n      coerce `tf.Graph`, `tf.compat.v1.GraphDef`, and string types.\n    step: The global step variable. This doesn't have useful semantics\n      for graph summaries, but is used anyway, due to the structure of\n      event log files. This defaults to the global step.\n    name: A name for the operation (optional).\n\n  Returns:\n    The created `tf.Operation` or a `tf.no_op` if summary writing has\n    not been enabled for this context.\n\n  Raises:\n    TypeError: If `param` isn't already a `tf.Tensor` in graph mode.\n  "
    if not context.executing_eagerly() and (not isinstance(param, tensor_lib.Tensor)):
        raise TypeError(f'graph() needs a argument `param` to be tf.Tensor (e.g. tf.placeholder) in graph mode, but received param={param} of type {type(param).__name__}.')
    writer = _summary_state.writer
    if writer is None:
        return control_flow_ops.no_op()
    with ops.device('cpu:0'):
        if isinstance(param, (ops.Graph, graph_pb2.GraphDef)):
            tensor = ops.convert_to_tensor(_serialize_graph(param), dtypes.string)
        else:
            tensor = array_ops.identity(param)
        return gen_summary_ops.write_graph_summary(writer._resource, _choose_step(step), tensor, name=name)

@tf_export('summary.graph', v1=[])
def graph(graph_data):
    if False:
        i = 10
        return i + 15
    'Writes a TensorFlow graph summary.\n\n  Write an instance of `tf.Graph` or `tf.compat.v1.GraphDef` as summary only\n  in an eager mode. Please prefer to use the trace APIs (`tf.summary.trace_on`,\n  `tf.summary.trace_off`, and `tf.summary.trace_export`) when using\n  `tf.function` which can automatically collect and record graphs from\n  executions.\n\n  Usage Example:\n  ```py\n  writer = tf.summary.create_file_writer("/tmp/mylogs")\n\n  @tf.function\n  def f():\n    x = constant_op.constant(2)\n    y = constant_op.constant(3)\n    return x**y\n\n  with writer.as_default():\n    tf.summary.graph(f.get_concrete_function().graph)\n\n  # Another example: in a very rare use case, when you are dealing with a TF v1\n  # graph.\n  graph = tf.Graph()\n  with graph.as_default():\n    c = tf.constant(30.0)\n  with writer.as_default():\n    tf.summary.graph(graph)\n  ```\n\n  Args:\n    graph_data: The TensorFlow graph to write, as a `tf.Graph` or a\n      `tf.compat.v1.GraphDef`.\n\n  Returns:\n    True on success, or False if no summary was written because no default\n    summary writer was available.\n\n  Raises:\n    ValueError: `graph` summary API is invoked in a graph mode.\n  '
    if not context.executing_eagerly():
        raise ValueError('graph() cannot be invoked inside a graph context.')
    writer = _summary_state.writer
    if writer is None:
        return constant_op.constant(False)
    with ops.device('cpu:0'):
        if not should_record_summaries():
            return constant_op.constant(False)
        if isinstance(graph_data, (ops.Graph, graph_pb2.GraphDef)):
            tensor = ops.convert_to_tensor(_serialize_graph(graph_data), dtypes.string)
        else:
            raise ValueError(f"Argument 'graph_data' is not tf.Graph or tf.compat.v1.GraphDef. Received graph_data={graph_data} of type {type(graph_data).__name__}.")
        gen_summary_ops.write_graph_summary(writer._resource, 0, tensor)
        return constant_op.constant(True)

def import_event(tensor, name=None):
    if False:
        print('Hello World!')
    'Writes a `tf.compat.v1.Event` binary proto.\n\n  This can be used to import existing event logs into a new summary writer sink.\n  Please note that this is lower level than the other summary functions and\n  will ignore the `tf.summary.should_record_summaries` setting.\n\n  Args:\n    tensor: A `tf.Tensor` of type `string` containing a serialized\n      `tf.compat.v1.Event` proto.\n    name: A name for the operation (optional).\n\n  Returns:\n    The created `tf.Operation`.\n  '
    return gen_summary_ops.import_event(_summary_state.writer._resource, tensor, name=name)

@tf_export('summary.flush', v1=[])
def flush(writer=None, name=None):
    if False:
        return 10
    'Forces summary writer to send any buffered data to storage.\n\n  This operation blocks until that finishes.\n\n  Args:\n    writer: The `tf.summary.SummaryWriter` to flush. If None, the current\n      default writer will be used instead; if there is no current writer, this\n      returns `tf.no_op`.\n    name: Ignored legacy argument for a name for the operation.\n\n  Returns:\n    The created `tf.Operation`.\n  '
    del name
    if writer is None:
        writer = _summary_state.writer
        if writer is None:
            return control_flow_ops.no_op()
    if isinstance(writer, SummaryWriter):
        return writer.flush()
    raise ValueError('Invalid argument to flush(): %r' % (writer,))

def legacy_raw_flush(writer=None, name=None):
    if False:
        i = 10
        return i + 15
    'Legacy version of flush() that accepts a raw resource tensor for `writer`.\n\n  Do not use this function in any new code. Not supported and not part of the\n  public TF APIs.\n\n  Args:\n    writer: The `tf.summary.SummaryWriter` to flush. If None, the current\n      default writer will be used instead; if there is no current writer, this\n      returns `tf.no_op`. For this legacy version only, also accepts a raw\n      resource tensor pointing to the underlying C++ writer resource.\n    name: Ignored legacy argument for a name for the operation.\n\n  Returns:\n    The created `tf.Operation`.\n  '
    if writer is None or isinstance(writer, SummaryWriter):
        return flush(writer, name)
    else:
        with ops.device('cpu:0'):
            return gen_summary_ops.flush_summary_writer(writer, name=name)

def eval_dir(model_dir, name=None):
    if False:
        return 10
    'Construct a logdir for an eval summary writer.'
    return os.path.join(model_dir, 'eval' if not name else 'eval_' + name)

@deprecation.deprecated(date=None, instructions='Renamed to create_file_writer().')
def create_summary_file_writer(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Please use `tf.contrib.summary.create_file_writer`.'
    logging.warning('Deprecation Warning: create_summary_file_writer was renamed to create_file_writer')
    return create_file_writer(*args, **kwargs)

def _serialize_graph(arbitrary_graph):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(arbitrary_graph, ops.Graph):
        return arbitrary_graph.as_graph_def(add_shapes=True).SerializeToString()
    else:
        return arbitrary_graph.SerializeToString()

def _choose_step(step):
    if False:
        while True:
            i = 10
    if step is None:
        return training_util.get_or_create_global_step()
    if not isinstance(step, tensor_lib.Tensor):
        return ops.convert_to_tensor(step, dtypes.int64)
    return step

def _check_create_file_writer_args(inside_function, **kwargs):
    if False:
        return 10
    'Helper to check the validity of arguments to a create_file_writer() call.\n\n  Args:\n    inside_function: whether the create_file_writer() call is in a tf.function\n    **kwargs: the arguments to check, as kwargs to give them names.\n\n  Raises:\n    ValueError: if the arguments are graph tensors.\n  '
    for (arg_name, arg) in kwargs.items():
        if not isinstance(arg, ops.EagerTensor) and tensor_util.is_tf_type(arg):
            if inside_function:
                raise ValueError(f"Invalid graph Tensor argument '{arg_name}={arg}' to create_file_writer() inside an @tf.function. The create call will be lifted into the outer eager execution context, so it cannot consume graph tensors defined inside the function body.")
            else:
                raise ValueError(f"Invalid graph Tensor argument '{arg_name}={arg}' to eagerly executed create_file_writer().")

def run_metadata(name, data, step=None):
    if False:
        print('Hello World!')
    'Writes entire RunMetadata summary.\n\n  A RunMetadata can contain DeviceStats, partition graphs, and function graphs.\n  Please refer to the proto for definition of each field.\n\n  Args:\n    name: A name for this summary. The summary tag used for TensorBoard will be\n      this name prefixed by any active name scopes.\n    data: A RunMetadata proto to write.\n    step: Explicit `int64`-castable monotonic step value for this summary. If\n      omitted, this defaults to `tf.summary.experimental.get_step()`, which must\n      not be None.\n\n  Returns:\n    True on success, or false if no summary was written because no default\n    summary writer was available.\n\n  Raises:\n    ValueError: if a default writer exists, but no step was provided and\n      `tf.summary.experimental.get_step()` is None.\n  '
    summary_metadata = summary_pb2.SummaryMetadata()
    summary_metadata.plugin_data.plugin_name = 'graph_run_metadata'
    summary_metadata.plugin_data.content = b'1'
    with summary_scope(name, 'graph_run_metadata_summary', [data, step]) as (tag, _):
        with ops.device('cpu:0'):
            tensor = constant_op.constant(data.SerializeToString(), dtype=dtypes.string)
        return write(tag=tag, tensor=tensor, step=step, metadata=summary_metadata)

def run_metadata_graphs(name, data, step=None):
    if False:
        while True:
            i = 10
    'Writes graphs from a RunMetadata summary.\n\n  Args:\n    name: A name for this summary. The summary tag used for TensorBoard will be\n      this name prefixed by any active name scopes.\n    data: A RunMetadata proto to write.\n    step: Explicit `int64`-castable monotonic step value for this summary. If\n      omitted, this defaults to `tf.summary.experimental.get_step()`, which must\n      not be None.\n\n  Returns:\n    True on success, or false if no summary was written because no default\n    summary writer was available.\n\n  Raises:\n    ValueError: if a default writer exists, but no step was provided and\n      `tf.summary.experimental.get_step()` is None.\n  '
    summary_metadata = summary_pb2.SummaryMetadata()
    summary_metadata.plugin_data.plugin_name = 'graph_run_metadata_graph'
    summary_metadata.plugin_data.content = b'1'
    data = config_pb2.RunMetadata(function_graphs=data.function_graphs, partition_graphs=data.partition_graphs)
    with summary_scope(name, 'graph_run_metadata_graph_summary', [data, step]) as (tag, _):
        with ops.device('cpu:0'):
            tensor = constant_op.constant(data.SerializeToString(), dtype=dtypes.string)
        return write(tag=tag, tensor=tensor, step=step, metadata=summary_metadata)
_TraceContext = collections.namedtuple('TraceContext', ('graph', 'profiler'))
_current_trace_context_lock = threading.Lock()
_current_trace_context = None

@tf_export('summary.trace_on', v1=[])
def trace_on(graph=True, profiler=False):
    if False:
        return 10
    'Starts a trace to record computation graphs and profiling information.\n\n  Must be invoked in eager mode.\n\n  When enabled, TensorFlow runtime will collect information that can later be\n  exported and consumed by TensorBoard. The trace is activated across the entire\n  TensorFlow runtime and affects all threads of execution.\n\n  To stop the trace and export the collected information, use\n  `tf.summary.trace_export`. To stop the trace without exporting, use\n  `tf.summary.trace_off`.\n\n  Args:\n    graph: If True, enables collection of executed graphs. It includes ones from\n        tf.function invocation and ones from the legacy graph mode. The default\n        is True.\n    profiler: If True, enables the advanced profiler. Enabling profiler\n        implicitly enables the graph collection. The profiler may incur a high\n        memory overhead. The default is False.\n\n  '
    if ops.inside_function():
        logging.warn('Cannot enable trace inside a tf.function.')
        return
    if not context.executing_eagerly():
        logging.warn('Must enable trace in eager mode.')
        return
    global _current_trace_context
    with _current_trace_context_lock:
        if _current_trace_context:
            logging.warn('Trace already enabled')
            return
        if graph and (not profiler):
            context.context().enable_graph_collection()
        if profiler:
            context.context().enable_run_metadata()
            _profiler.start()
        _current_trace_context = _TraceContext(graph=graph, profiler=profiler)

@tf_export('summary.trace_export', v1=[])
def trace_export(name, step=None, profiler_outdir=None):
    if False:
        print('Hello World!')
    'Stops and exports the active trace as a Summary and/or profile file.\n\n  Stops the trace and exports all metadata collected during the trace to the\n  default SummaryWriter, if one has been set.\n\n  Args:\n    name: A name for the summary to be written.\n    step: Explicit `int64`-castable monotonic step value for this summary. If\n      omitted, this defaults to `tf.summary.experimental.get_step()`, which must\n      not be None.\n    profiler_outdir: Output directory for profiler. It is required when profiler\n      is enabled when trace was started. Otherwise, it is ignored.\n\n  Raises:\n    ValueError: if a default writer exists, but no step was provided and\n      `tf.summary.experimental.get_step()` is None.\n  '
    global _current_trace_context
    if ops.inside_function():
        logging.warn('Cannot export trace inside a tf.function.')
        return
    if not context.executing_eagerly():
        logging.warn('Can only export trace while executing eagerly.')
        return
    with _current_trace_context_lock:
        if _current_trace_context is None:
            raise ValueError('Must enable trace before export through tf.summary.trace_on.')
        (graph, profiler) = _current_trace_context
        if profiler and profiler_outdir is None:
            raise ValueError('Argument `profiler_outdir` is not specified.')
    run_meta = context.context().export_run_metadata()
    if graph and (not profiler):
        run_metadata_graphs(name, run_meta, step)
    else:
        run_metadata(name, run_meta, step)
    if profiler:
        _profiler.save(profiler_outdir, _profiler.stop())
    trace_off()

@tf_export('summary.trace_off', v1=[])
def trace_off():
    if False:
        return 10
    'Stops the current trace and discards any collected information.'
    global _current_trace_context
    with _current_trace_context_lock:
        if _current_trace_context is None:
            return
        (graph, profiler) = _current_trace_context
        _current_trace_context = None
    if graph:
        context.context().disable_run_metadata()
    if profiler:
        try:
            _profiler.stop()
        except _profiler.ProfilerNotRunningError:
            pass

def _maybe_convert_tensor_to_dtensor(writer, tensor):
    if False:
        while True:
            i = 10
    if getattr(writer, '_mesh', None) is not None:
        mesh = writer._mesh.host_mesh()
        tensor = dtensor_api.copy_to_mesh(tensor, layout_lib.Layout.replicated(mesh, rank=tensor.shape.rank))
    return tensor