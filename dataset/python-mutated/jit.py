"""Library for controlling the Tensorflow/XLA JIT compiler."""
import contextlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export
_XLA_SCOPE_KEY = ('__xla_scope',)

class _XlaScope(object):
    """Keeps track of previous XLA scope calls, and depth of current call."""

    def __init__(self, count, depth):
        if False:
            i = 10
            return i + 15
        self.count = count
        self.depth = depth

@contextlib.contextmanager
@tf_export('xla.experimental.jit_scope')
def experimental_jit_scope(compile_ops=True, separate_compiled_gradients=False):
    if False:
        while True:
            i = 10
    "Enable or disable JIT compilation of operators within the scope.\n\n  NOTE: This is an experimental feature.\n\n  The compilation is a hint and only supported on a best-effort basis.\n\n  Example usage:\n\n    ```python\n    with tf.xla.experimental.jit_scope():\n      c = tf.matmul(a, b)  # compiled\n    with tf.xla.experimental.jit_scope(compile_ops=False):\n      d = tf.matmul(a, c)  # not compiled\n    with tf.xla.experimental.jit_scope(\n        compile_ops=lambda node_def: 'matmul' in node_def.op.lower()):\n      e = tf.matmul(a, b) + d  # matmul is compiled, the addition is not.\n    ```\n\n  Example of `separate_compiled_gradients`:\n\n    ```python\n    # In the example below, the computations for f, g and h will all be compiled\n    # in separate scopes.\n    with tf.xla.experimental.jit_scope(\n        separate_compiled_gradients=True):\n      f = tf.matmul(a, b)\n    g = tf.gradients([f], [a, b], name='mygrads1')\n    h = tf.gradients([f], [a, b], name='mygrads2')\n    ```\n\n  Ops that are not in the scope may be clustered and compiled with ops in\n  the scope with `compile_ops=True`, while the ops in the scope with\n  `compile_ops=False` will never be compiled.\n\n  For example:\n\n    ```python\n    # In the example below, x and loss may be clustered and compiled together,\n    # while y will not be compiled.\n    with tf.xla.experimental.jit_scope():\n      x = tf.matmul(a, b)\n    with tf.xla.experimental.jit_scope(compile_ops=False):\n      y = tf.matmul(c, d)\n    loss = x + y\n    ```\n\n  If you want to only compile the ops in the scope with `compile_ops=True`,\n  consider adding an outer `jit_scope(compile_ops=False)`:\n\n    ```python\n    # In the example below, only x will be compiled.\n    with tf.xla.experimental.jit_scope(compile_ops=False):\n      with tf.xla.experimental.jit_scope():\n        x = tf.matmul(a, b)\n      y = tf.matmul(c, d)\n      loss = x + y\n    ```\n\n  Args:\n    compile_ops: Whether to enable or disable compilation in the scope.\n      Either a Python bool, or a callable that accepts the parameter\n      `node_def` and returns a python bool.\n    separate_compiled_gradients: If true put each gradient subgraph into a\n      separate compilation scope. This gives fine-grained control over which\n      portions of the graph will be compiled as a single unit. Compiling\n      gradients separately may yield better performance for some graphs.\n      The scope is named based on the scope of the forward computation as well\n      as the name of the gradients. As a result, the gradients will be compiled\n      in a scope that is separate from both the forward computation, and from\n      other gradients.\n  Raises:\n    RuntimeError: if called when eager execution is enabled.\n  Yields:\n    The current scope, enabling or disabling compilation.\n  "
    if context.executing_eagerly():
        raise RuntimeError('xla.experimental.jit_scope is not supported when eager execution is enabled. Try use it inside tf.function.')
    if callable(compile_ops):

        def xla_compile(node_def):
            if False:
                i = 10
                return i + 15
            return attr_value_pb2.AttrValue(b=compile_ops(node_def))
    else:
        xla_compile = attr_value_pb2.AttrValue(b=compile_ops)
    attrs = {'_XlaCompile': xla_compile, '_XlaSeparateCompiledGradients': attr_value_pb2.AttrValue(b=bool(separate_compiled_gradients))}
    xla_scope_counter = ops.get_collection(_XLA_SCOPE_KEY)
    if not xla_scope_counter:
        xla_scope_counter = _XlaScope(0, 0)
        ops.add_to_collection(_XLA_SCOPE_KEY, xla_scope_counter)
    else:
        xla_scope_counter = xla_scope_counter[0]
    if xla_scope_counter.depth == 0:
        attrs['_XlaScope'] = attr_value_pb2.AttrValue(s=('jit_scope_%d' % xla_scope_counter.count).encode())
        xla_scope_counter.count += 1
    xla_scope_counter.depth += 1
    with ops.get_default_graph()._attr_scope(attrs):
        yield
    xla_scope_counter.depth -= 1