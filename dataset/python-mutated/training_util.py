"""Utility functions for training."""
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import cond
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
GLOBAL_STEP_READ_KEY = 'global_step_read_op_cache'
write_graph = graph_io.write_graph

@tf_export(v1=['train.global_step'])
def global_step(sess, global_step_tensor):
    if False:
        for i in range(10):
            print('nop')
    "Small helper to get the global step.\n\n  ```python\n  # Create a variable to hold the global_step.\n  global_step_tensor = tf.Variable(10, trainable=False, name='global_step')\n  # Create a session.\n  sess = tf.compat.v1.Session()\n  # Initialize the variable\n  sess.run(global_step_tensor.initializer)\n  # Get the variable value.\n  print('global_step: %s' % tf.compat.v1.train.global_step(sess,\n  global_step_tensor))\n\n  global_step: 10\n  ```\n\n  Args:\n    sess: A TensorFlow `Session` object.\n    global_step_tensor:  `Tensor` or the `name` of the operation that contains\n      the global step.\n\n  Returns:\n    The global step value.\n  "
    if context.executing_eagerly():
        return int(global_step_tensor.numpy())
    return int(sess.run(global_step_tensor))

@tf_export(v1=['train.get_global_step'])
def get_global_step(graph=None):
    if False:
        for i in range(10):
            print('nop')
    'Get the global step tensor.\n\n  The global step tensor must be an integer variable. We first try to find it\n  in the collection `GLOBAL_STEP`, or by name `global_step:0`.\n\n  Args:\n    graph: The graph to find the global step in. If missing, use default graph.\n\n  Returns:\n    The global step variable, or `None` if none was found.\n\n  Raises:\n    TypeError: If the global step tensor has a non-integer type, or if it is not\n      a `Variable`.\n\n  @compatibility(TF2)\n  With the deprecation of global graphs, TF no longer tracks variables in\n  collections. In other words, there are no global variables in TF2. Thus, the\n  global step functions have been removed  (`get_or_create_global_step`,\n  `create_global_step`, `get_global_step`) . You have two options for migrating:\n\n  1. Create a Keras optimizer, which generates an `iterations` variable. This\n     variable is automatically incremented when calling `apply_gradients`.\n  2. Manually create and increment a `tf.Variable`.\n\n  Below is an example of migrating away from using a global step to using a\n  Keras optimizer:\n\n  Define a dummy model and loss:\n\n  >>> def compute_loss(x):\n  ...   v = tf.Variable(3.0)\n  ...   y = x * v\n  ...   loss = x * 5 - x * v\n  ...   return loss, [v]\n\n  Before migrating:\n\n  >>> g = tf.Graph()\n  >>> with g.as_default():\n  ...   x = tf.compat.v1.placeholder(tf.float32, [])\n  ...   loss, var_list = compute_loss(x)\n  ...   global_step = tf.compat.v1.train.get_or_create_global_step()\n  ...   global_init = tf.compat.v1.global_variables_initializer()\n  ...   optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)\n  ...   train_op = optimizer.minimize(loss, global_step, var_list)\n  >>> sess = tf.compat.v1.Session(graph=g)\n  >>> sess.run(global_init)\n  >>> print("before training:", sess.run(global_step))\n  before training: 0\n  >>> sess.run(train_op, feed_dict={x: 3})\n  >>> print("after training:", sess.run(global_step))\n  after training: 1\n\n  Using `get_global_step`:\n\n  >>> with g.as_default():\n  ...   print(sess.run(tf.compat.v1.train.get_global_step()))\n  1\n\n  Migrating to a Keras optimizer:\n\n  >>> optimizer = tf.keras.optimizers.SGD(.01)\n  >>> print("before training:", optimizer.iterations.numpy())\n  before training: 0\n  >>> with tf.GradientTape() as tape:\n  ...   loss, var_list = compute_loss(3)\n  ...   grads = tape.gradient(loss, var_list)\n  ...   optimizer.apply_gradients(zip(grads, var_list))\n  >>> print("after training:", optimizer.iterations.numpy())\n  after training: 1\n\n  @end_compatibility\n  '
    graph = graph or ops.get_default_graph()
    global_step_tensor = None
    global_step_tensors = graph.get_collection(ops.GraphKeys.GLOBAL_STEP)
    if len(global_step_tensors) == 1:
        global_step_tensor = global_step_tensors[0]
    elif not global_step_tensors:
        try:
            global_step_tensor = graph.get_tensor_by_name('global_step:0')
        except KeyError:
            return None
    else:
        logging.error('Multiple tensors in global_step collection.')
        return None
    assert_global_step(global_step_tensor)
    return global_step_tensor

@tf_export(v1=['train.create_global_step'])
def create_global_step(graph=None):
    if False:
        i = 10
        return i + 15
    'Create global step tensor in graph.\n\n  Args:\n    graph: The graph in which to create the global step tensor. If missing, use\n      default graph.\n\n  Returns:\n    Global step tensor.\n\n  Raises:\n    ValueError: if global step tensor is already defined.\n\n  @compatibility(TF2)\n  With the deprecation of global graphs, TF no longer tracks variables in\n  collections. In other words, there are no global variables in TF2. Thus, the\n  global step functions have been removed  (`get_or_create_global_step`,\n  `create_global_step`, `get_global_step`) . You have two options for migrating:\n\n  1. Create a Keras optimizer, which generates an `iterations` variable. This\n     variable is automatically incremented when calling `apply_gradients`.\n  2. Manually create and increment a `tf.Variable`.\n\n  Below is an example of migrating away from using a global step to using a\n  Keras optimizer:\n\n  Define a dummy model and loss:\n\n  >>> def compute_loss(x):\n  ...   v = tf.Variable(3.0)\n  ...   y = x * v\n  ...   loss = x * 5 - x * v\n  ...   return loss, [v]\n\n  Before migrating:\n\n  >>> g = tf.Graph()\n  >>> with g.as_default():\n  ...   x = tf.compat.v1.placeholder(tf.float32, [])\n  ...   loss, var_list = compute_loss(x)\n  ...   global_step = tf.compat.v1.train.create_global_step()\n  ...   global_init = tf.compat.v1.global_variables_initializer()\n  ...   optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)\n  ...   train_op = optimizer.minimize(loss, global_step, var_list)\n  >>> sess = tf.compat.v1.Session(graph=g)\n  >>> sess.run(global_init)\n  >>> print("before training:", sess.run(global_step))\n  before training: 0\n  >>> sess.run(train_op, feed_dict={x: 3})\n  >>> print("after training:", sess.run(global_step))\n  after training: 1\n\n  Migrating to a Keras optimizer:\n\n  >>> optimizer = tf.keras.optimizers.SGD(.01)\n  >>> print("before training:", optimizer.iterations.numpy())\n  before training: 0\n  >>> with tf.GradientTape() as tape:\n  ...   loss, var_list = compute_loss(3)\n  ...   grads = tape.gradient(loss, var_list)\n  ...   optimizer.apply_gradients(zip(grads, var_list))\n  >>> print("after training:", optimizer.iterations.numpy())\n  after training: 1\n\n  @end_compatibility\n  '
    graph = graph or ops.get_default_graph()
    if get_global_step(graph) is not None:
        raise ValueError('"global_step" already exists.')
    if context.executing_eagerly():
        with ops.device('cpu:0'):
            return variable_scope.get_variable(ops.GraphKeys.GLOBAL_STEP, shape=[], dtype=dtypes.int64, initializer=init_ops.zeros_initializer(), trainable=False, aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA, collections=[ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.GLOBAL_STEP])
    with graph.as_default() as g, g.name_scope(None):
        return variable_scope.get_variable(ops.GraphKeys.GLOBAL_STEP, shape=[], dtype=dtypes.int64, initializer=init_ops.zeros_initializer(), trainable=False, aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA, collections=[ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.GLOBAL_STEP])

@tf_export(v1=['train.get_or_create_global_step'])
def get_or_create_global_step(graph=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns and create (if necessary) the global step tensor.\n\n  Args:\n    graph: The graph in which to create the global step tensor. If missing, use\n      default graph.\n\n  Returns:\n    The global step tensor.\n\n  @compatibility(TF2)\n  With the deprecation of global graphs, TF no longer tracks variables in\n  collections. In other words, there are no global variables in TF2. Thus, the\n  global step functions have been removed  (`get_or_create_global_step`,\n  `create_global_step`, `get_global_step`) . You have two options for migrating:\n\n  1. Create a Keras optimizer, which generates an `iterations` variable. This\n     variable is automatically incremented when calling `apply_gradients`.\n  2. Manually create and increment a `tf.Variable`.\n\n  Below is an example of migrating away from using a global step to using a\n  Keras optimizer:\n\n  Define a dummy model and loss:\n\n  >>> def compute_loss(x):\n  ...   v = tf.Variable(3.0)\n  ...   y = x * v\n  ...   loss = x * 5 - x * v\n  ...   return loss, [v]\n\n  Before migrating:\n\n  >>> g = tf.Graph()\n  >>> with g.as_default():\n  ...   x = tf.compat.v1.placeholder(tf.float32, [])\n  ...   loss, var_list = compute_loss(x)\n  ...   global_step = tf.compat.v1.train.get_or_create_global_step()\n  ...   global_init = tf.compat.v1.global_variables_initializer()\n  ...   optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)\n  ...   train_op = optimizer.minimize(loss, global_step, var_list)\n  >>> sess = tf.compat.v1.Session(graph=g)\n  >>> sess.run(global_init)\n  >>> print("before training:", sess.run(global_step))\n  before training: 0\n  >>> sess.run(train_op, feed_dict={x: 3})\n  >>> print("after training:", sess.run(global_step))\n  after training: 1\n\n  Migrating to a Keras optimizer:\n\n  >>> optimizer = tf.keras.optimizers.SGD(.01)\n  >>> print("before training:", optimizer.iterations.numpy())\n  before training: 0\n  >>> with tf.GradientTape() as tape:\n  ...   loss, var_list = compute_loss(3)\n  ...   grads = tape.gradient(loss, var_list)\n  ...   optimizer.apply_gradients(zip(grads, var_list))\n  >>> print("after training:", optimizer.iterations.numpy())\n  after training: 1\n\n  @end_compatibility\n  '
    graph = graph or ops.get_default_graph()
    global_step_tensor = get_global_step(graph)
    if global_step_tensor is None:
        global_step_tensor = create_global_step(graph)
    return global_step_tensor

@tf_export(v1=['train.assert_global_step'])
def assert_global_step(global_step_tensor):
    if False:
        for i in range(10):
            print('nop')
    'Asserts `global_step_tensor` is a scalar int `Variable` or `Tensor`.\n\n  Args:\n    global_step_tensor: `Tensor` to test.\n  '
    if not (isinstance(global_step_tensor, variables.Variable) or isinstance(global_step_tensor, tensor.Tensor) or resource_variable_ops.is_resource_variable(global_step_tensor)):
        raise TypeError('Existing "global_step" must be a Variable or Tensor: %s.' % global_step_tensor)
    if not global_step_tensor.dtype.base_dtype.is_integer:
        raise TypeError('Existing "global_step" does not have integer type: %s' % global_step_tensor.dtype)
    if global_step_tensor.get_shape().ndims != 0 and global_step_tensor.get_shape().is_fully_defined():
        raise TypeError('Existing "global_step" is not scalar: %s' % global_step_tensor.get_shape())

def _get_global_step_read(graph=None):
    if False:
        return 10
    'Gets global step read tensor in graph.\n\n  Args:\n    graph: The graph in which to create the global step read tensor. If missing,\n      use default graph.\n\n  Returns:\n    Global step read tensor.\n\n  Raises:\n    RuntimeError: if multiple items found in collection GLOBAL_STEP_READ_KEY.\n  '
    graph = graph or ops.get_default_graph()
    global_step_read_tensors = graph.get_collection(GLOBAL_STEP_READ_KEY)
    if len(global_step_read_tensors) > 1:
        raise RuntimeError('There are multiple items in collection {}. There should be only one.'.format(GLOBAL_STEP_READ_KEY))
    if len(global_step_read_tensors) == 1:
        return global_step_read_tensors[0]
    return None

def _get_or_create_global_step_read(graph=None):
    if False:
        print('Hello World!')
    'Gets or creates global step read tensor in graph.\n\n  Args:\n    graph: The graph in which to create the global step read tensor. If missing,\n      use default graph.\n\n  Returns:\n    Global step read tensor if there is global_step_tensor else return None.\n  '
    graph = graph or ops.get_default_graph()
    global_step_read_tensor = _get_global_step_read(graph)
    if global_step_read_tensor is not None:
        return global_step_read_tensor
    global_step_tensor = get_global_step(graph)
    if global_step_tensor is None:
        return None
    with graph.as_default() as g, g.name_scope(None):
        with g.name_scope(global_step_tensor.op.name + '/'):
            if isinstance(global_step_tensor, variables.Variable):
                global_step_value = cond.cond(variable_v1.is_variable_initialized(global_step_tensor), global_step_tensor.read_value, lambda : global_step_tensor.initial_value)
            else:
                global_step_value = global_step_tensor
            global_step_read_tensor = global_step_value + 0
            ops.add_to_collection(GLOBAL_STEP_READ_KEY, global_step_read_tensor)
    return _get_global_step_read(graph)

def _increment_global_step(increment, graph=None):
    if False:
        for i in range(10):
            print('nop')
    graph = graph or ops.get_default_graph()
    global_step_tensor = get_global_step(graph)
    if global_step_tensor is None:
        raise ValueError('Global step tensor should be created by tf.train.get_or_create_global_step before calling increment.')
    global_step_read_tensor = _get_or_create_global_step_read(graph)
    with graph.as_default() as g, g.name_scope(None):
        with g.name_scope(global_step_tensor.op.name + '/'):
            with ops.control_dependencies([global_step_read_tensor]):
                return state_ops.assign_add(global_step_tensor, increment)