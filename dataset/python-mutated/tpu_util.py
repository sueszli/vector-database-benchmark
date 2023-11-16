"""Utility functions for TPU."""
import contextlib
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.tpu import tpu_replication

def enclosing_tpu_context():
    if False:
        while True:
            i = 10
    'Returns the TPUReplicateContext, which exists inside a tpu.rewrite().'
    return enclosing_tpu_context_and_graph()[0]

def enclosing_tpu_context_and_graph():
    if False:
        i = 10
        return i + 15
    'Returns the TPUReplicateContext which exists inside a tpu.rewrite(), and its associated graph.'
    graph = ops.get_default_graph()
    while graph is not None:
        ctx = graph._get_control_flow_context()
        while ctx is not None:
            if isinstance(ctx, tpu_replication.TPUReplicateContext):
                return (ctx, graph)
            ctx = ctx.outer_context
        graph = getattr(graph, 'outer_graph', None)
    return (None, None)

@contextlib.contextmanager
def outside_or_skip_tpu_context():
    if False:
        return 10
    'Returns a context manager that skips current enclosing context if there is any.'
    (ctx, graph) = enclosing_tpu_context_and_graph()
    if ctx is None:
        yield
    else:
        saved_context = graph._get_control_flow_context()
        graph._set_control_flow_context(ctx.outer_context)
        yield
        graph._set_control_flow_context(saved_context)

@contextlib.contextmanager
def _maybe_enter_graph(tensor):
    if False:
        return 10
    if context.executing_eagerly() or isinstance(tensor, ops.EagerTensor) or ops.has_default_graph():
        yield
    else:
        with tensor.graph.as_default():
            yield

@contextlib.contextmanager
def _maybe_on_device(var):
    if False:
        return 10
    if isinstance(var, packed.PackedVarAndDevice):
        with ops.device(var.device):
            yield
    else:
        yield

def make_raw_assign_fn(raw_assign_fn, use_handle=True):
    if False:
        for i in range(10):
            print('nop')
    'Wrap `raw_assign_fn` with the proper graph context and device scope.\n\n  Args:\n    raw_assign_fn: the function to be wrapped.\n    use_handle: if True, the `raw_assign_fn` will be applied to the handle of a\n      variable; otherwise it will be applied to the variable itself.\n\n  Returns:\n    The wrapped function.\n  '

    def assign_fn(var, value, use_locking=False, name=None, read_value=True):
        if False:
            print('Hello World!')
        del use_locking
        handle = var.handle if use_handle else var
        with _maybe_enter_graph(handle), _maybe_on_device(var):
            op = raw_assign_fn(handle, ops.convert_to_tensor(value, dtype=var.dtype), name=name)
            with ops.control_dependencies([op]):
                if read_value:
                    return var._read_variable_op() if use_handle else var.read_value()
                else:
                    return op
    return assign_fn

def make_raw_scatter_xxx_fn(raw_scatter_xxx_fn):
    if False:
        print('Hello World!')
    'Wrap `raw_scatter_xxx_fn` so that it can be called w/ and w/o packed handle.'

    def scatter_xxx_fn(var, sparse_delta, use_locking=False, name=None):
        if False:
            print('Hello World!')
        del use_locking
        handle = var.handle
        with _maybe_enter_graph(handle), _maybe_on_device(var):
            op = raw_scatter_xxx_fn(handle, sparse_delta.indices, ops.convert_to_tensor(sparse_delta.values, var.dtype), name=name)
            with ops.control_dependencies([op]):
                return var._read_variable_op()
    return scatter_xxx_fn

class LazyVariableTracker(object):
    """Class to track uninitialized lazy variables."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self._uninitialized_var_list = []

    def initialize_all(self):
        if False:
            return 10
        'Initialize all uninitialized lazy variables stored in scope.'

        def assign_function(uninitialized_var_list):
            if False:
                i = 10
                return i + 15
            for var in uninitialized_var_list:
                val = var._initial_value
                packed_var = getattr(var, '_packed_var', None)
                handle = getattr(packed_var, 'packed_handle', var.handle)
                with ops.device(handle.device):
                    resource_variable_ops.AssignVariableOp(resource=handle, value=val)
            return constant_op.constant([])
        assign_tf_function = def_function.function(assign_function, autograph=False, jit_compile=False)
        with ops.init_scope():
            if len(self._uninitialized_var_list) > 1:
                assign_tf_function(self._uninitialized_var_list)
            else:
                assign_function(self._uninitialized_var_list)
        self._uninitialized_var_list = []

    def add_uninitialized_var(self, var):
        if False:
            while True:
                i = 10
        self._uninitialized_var_list.append(var)

class TPUUninitializedVariable(resource_variable_ops.UninitializedVariable):
    """UninitializedVariable component for TPU.

  Sometimes user might assign (different values) to a single component of a
  mirrored TPU variable. Thus we need to initialize_all when the assign* or read
  is invoked on a single component.
  """

    def read_value(self):
        if False:
            print('Hello World!')
        self._lazy_scope.initialize_all()
        return super().read_value()

    def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
        if False:
            while True:
                i = 10
        self._lazy_scope.initialize_all()
        return super().assign_sub(delta, use_locking=use_locking, name=name, read_value=read_value)

    def assign(self, value, use_locking=None, name=None, read_value=True):
        if False:
            i = 10
            return i + 15
        self._lazy_scope.initialize_all()
        return super().assign(value, use_locking=use_locking, name=name, read_value=read_value)

    def assign_add(self, delta, use_locking=None, name=None, read_value=True):
        if False:
            return 10
        self._lazy_scope.initialize_all()
        return super().assign_add(delta, use_locking=use_locking, name=name, read_value=read_value)