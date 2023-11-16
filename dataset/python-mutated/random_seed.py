"""For seeding individual ops based on a graph-level seed.
"""
import weakref
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
DEFAULT_GRAPH_SEED = 87654321
_MAXINT32 = 2 ** 31 - 1
_graph_to_seed_dict = weakref.WeakKeyDictionary()

def _truncate_seed(seed):
    if False:
        i = 10
        return i + 15
    return seed % _MAXINT32

@tf_export(v1=['random.get_seed', 'get_seed'])
@deprecation.deprecated_endpoints('get_seed')
def get_seed(op_seed):
    if False:
        for i in range(10):
            print('nop')
    'Returns the local seeds an operation should use given an op-specific seed.\n\n  Given operation-specific seed, `op_seed`, this helper function returns two\n  seeds derived from graph-level and op-level seeds. Many random operations\n  internally use the two seeds to allow user to change the seed globally for a\n  graph, or for only specific operations.\n\n  For details on how the graph-level seed interacts with op seeds, see\n  `tf.compat.v1.random.set_random_seed`.\n\n  Args:\n    op_seed: integer.\n\n  Returns:\n    A tuple of two integers that should be used for the local seed of this\n    operation.\n  '
    eager = context.executing_eagerly()
    if eager:
        global_seed = context.global_seed()
    else:
        global_seed = ops.get_default_graph().seed
    if global_seed is not None:
        if op_seed is None:
            if hasattr(ops.get_default_graph(), '_seed_used'):
                ops.get_default_graph()._seed_used = True
            if eager:
                op_seed = context.internal_operation_seed()
            else:
                op_seed = _graph_to_seed_dict.setdefault(ops.get_default_graph(), 0)
                _graph_to_seed_dict[ops.get_default_graph()] += 1
        seeds = (_truncate_seed(global_seed), _truncate_seed(op_seed))
    elif op_seed is not None:
        seeds = (DEFAULT_GRAPH_SEED, _truncate_seed(op_seed))
    else:
        seeds = (None, None)
    if seeds == (None, None) and config.is_op_determinism_enabled():
        raise RuntimeError('Random ops require a seed to be set when determinism is enabled. Please set a seed before running the op, e.g. by calling tf.random.set_seed(1).')
    if seeds == (0, 0):
        return (0, _MAXINT32)
    return seeds

@tf_export(v1=['random.set_random_seed', 'set_random_seed'])
def set_random_seed(seed):
    if False:
        while True:
            i = 10
    'Sets the graph-level random seed for the default graph.\n\n  Operations that rely on a random seed actually derive it from two seeds:\n  the graph-level and operation-level seeds. This sets the graph-level seed.\n\n  Its interactions with operation-level seeds is as follows:\n\n    1. If neither the graph-level nor the operation seed is set:\n      A random seed is used for this op.\n    2. If the graph-level seed is set, but the operation seed is not:\n      The system deterministically picks an operation seed in conjunction with\n      the graph-level seed so that it gets a unique random sequence. Within the\n      same version of tensorflow and user code, this sequence is deterministic.\n      However across different versions, this sequence might change. If the\n      code depends on particular seeds to work, specify both graph-level\n      and operation-level seeds explicitly.\n    3. If the graph-level seed is not set, but the operation seed is set:\n      A default graph-level seed and the specified operation seed are used to\n      determine the random sequence.\n    4. If both the graph-level and the operation seed are set:\n      Both seeds are used in conjunction to determine the random sequence.\n\n  To illustrate the user-visible effects, consider these examples:\n\n  To generate different sequences across sessions, set neither\n  graph-level nor op-level seeds:\n\n  ```python\n  a = tf.random.uniform([1])\n  b = tf.random.normal([1])\n\n  print("Session 1")\n  with tf.compat.v1.Session() as sess1:\n    print(sess1.run(a))  # generates \'A1\'\n    print(sess1.run(a))  # generates \'A2\'\n    print(sess1.run(b))  # generates \'B1\'\n    print(sess1.run(b))  # generates \'B2\'\n\n  print("Session 2")\n  with tf.compat.v1.Session() as sess2:\n    print(sess2.run(a))  # generates \'A3\'\n    print(sess2.run(a))  # generates \'A4\'\n    print(sess2.run(b))  # generates \'B3\'\n    print(sess2.run(b))  # generates \'B4\'\n  ```\n\n  To generate the same repeatable sequence for an op across sessions, set the\n  seed for the op:\n\n  ```python\n  a = tf.random.uniform([1], seed=1)\n  b = tf.random.normal([1])\n\n  # Repeatedly running this block with the same graph will generate the same\n  # sequence of values for \'a\', but different sequences of values for \'b\'.\n  print("Session 1")\n  with tf.compat.v1.Session() as sess1:\n    print(sess1.run(a))  # generates \'A1\'\n    print(sess1.run(a))  # generates \'A2\'\n    print(sess1.run(b))  # generates \'B1\'\n    print(sess1.run(b))  # generates \'B2\'\n\n  print("Session 2")\n  with tf.compat.v1.Session() as sess2:\n    print(sess2.run(a))  # generates \'A1\'\n    print(sess2.run(a))  # generates \'A2\'\n    print(sess2.run(b))  # generates \'B3\'\n    print(sess2.run(b))  # generates \'B4\'\n  ```\n\n  To make the random sequences generated by all ops be repeatable across\n  sessions, set a graph-level seed:\n\n  ```python\n  tf.compat.v1.random.set_random_seed(1234)\n  a = tf.random.uniform([1])\n  b = tf.random.normal([1])\n\n  # Repeatedly running this block with the same graph will generate the same\n  # sequences of \'a\' and \'b\'.\n  print("Session 1")\n  with tf.compat.v1.Session() as sess1:\n    print(sess1.run(a))  # generates \'A1\'\n    print(sess1.run(a))  # generates \'A2\'\n    print(sess1.run(b))  # generates \'B1\'\n    print(sess1.run(b))  # generates \'B2\'\n\n  print("Session 2")\n  with tf.compat.v1.Session() as sess2:\n    print(sess2.run(a))  # generates \'A1\'\n    print(sess2.run(a))  # generates \'A2\'\n    print(sess2.run(b))  # generates \'B1\'\n    print(sess2.run(b))  # generates \'B2\'\n  ```\n\n  @compatibility(TF2)\n  \'tf.compat.v1.set_random_seed\' is compatible with eager mode. However,\n  in eager mode this API will set the global seed instead of the\n  graph-level seed of the default graph. In TF2 this API is changed to\n  [tf.random.set_seed]\n  (https://www.tensorflow.org/api_docs/python/tf/random/set_seed).\n  @end_compatibility\n\n  Args:\n    seed: integer.\n  '
    if context.executing_eagerly():
        context.set_global_seed(seed)
    else:
        ops.get_default_graph().seed = seed

@tf_export('random.set_seed', v1=[])
def set_seed(seed):
    if False:
        for i in range(10):
            print('nop')
    "Sets the global random seed.\n\n  Operations that rely on a random seed actually derive it from two seeds:\n  the global and operation-level seeds. This sets the global seed.\n\n  Its interactions with operation-level seeds is as follows:\n\n    1. If neither the global seed nor the operation seed is set: A randomly\n      picked seed is used for this op.\n    2. If the global seed is set, but the operation seed is not:\n      The system deterministically picks an operation seed in conjunction with\n      the global seed so that it gets a unique random sequence. Within the\n      same version of tensorflow and user code, this sequence is deterministic.\n      However across different versions, this sequence might change. If the\n      code depends on particular seeds to work, specify both global\n      and operation-level seeds explicitly.\n    3. If the operation seed is set, but the global seed is not set:\n      A default global seed and the specified operation seed are used to\n      determine the random sequence.\n    4. If both the global and the operation seed are set:\n      Both seeds are used in conjunction to determine the random sequence.\n\n  To illustrate the user-visible effects, consider these examples:\n\n  If neither the global seed nor the operation seed is set, we get different\n  results for every call to the random op and every re-run of the program:\n\n  ```python\n  print(tf.random.uniform([1]))  # generates 'A1'\n  print(tf.random.uniform([1]))  # generates 'A2'\n  ```\n\n  (now close the program and run it again)\n\n  ```python\n  print(tf.random.uniform([1]))  # generates 'A3'\n  print(tf.random.uniform([1]))  # generates 'A4'\n  ```\n\n  If the global seed is set but the operation seed is not set, we get different\n  results for every call to the random op, but the same sequence for every\n  re-run of the program:\n\n  ```python\n  tf.random.set_seed(1234)\n  print(tf.random.uniform([1]))  # generates 'A1'\n  print(tf.random.uniform([1]))  # generates 'A2'\n  ```\n\n  (now close the program and run it again)\n\n  ```python\n  tf.random.set_seed(1234)\n  print(tf.random.uniform([1]))  # generates 'A1'\n  print(tf.random.uniform([1]))  # generates 'A2'\n  ```\n\n  The reason we get 'A2' instead 'A1' on the second call of `tf.random.uniform`\n  above is because the second call uses a different operation seed.\n\n  Note that `tf.function` acts like a re-run of a program in this case. When\n  the global seed is set but operation seeds are not set, the sequence of random\n  numbers are the same for each `tf.function`. For example:\n\n  ```python\n  tf.random.set_seed(1234)\n\n  @tf.function\n  def f():\n    a = tf.random.uniform([1])\n    b = tf.random.uniform([1])\n    return a, b\n\n  @tf.function\n  def g():\n    a = tf.random.uniform([1])\n    b = tf.random.uniform([1])\n    return a, b\n\n  print(f())  # prints '(A1, A2)'\n  print(g())  # prints '(A1, A2)'\n  ```\n\n  If the operation seed is set, we get different results for every call to the\n  random op, but the same sequence for every re-run of the program:\n\n  ```python\n  print(tf.random.uniform([1], seed=1))  # generates 'A1'\n  print(tf.random.uniform([1], seed=1))  # generates 'A2'\n  ```\n\n  (now close the program and run it again)\n\n  ```python\n  print(tf.random.uniform([1], seed=1))  # generates 'A1'\n  print(tf.random.uniform([1], seed=1))  # generates 'A2'\n  ```\n\n  The reason we get 'A2' instead 'A1' on the second call of `tf.random.uniform`\n  above is because the same `tf.random.uniform` kernel (i.e. internal\n  representation) is used by TensorFlow for all calls of it with the same\n  arguments, and the kernel maintains an internal counter which is incremented\n  every time it is executed, generating different results.\n\n  Calling `tf.random.set_seed` will reset any such counters:\n\n  ```python\n  tf.random.set_seed(1234)\n  print(tf.random.uniform([1], seed=1))  # generates 'A1'\n  print(tf.random.uniform([1], seed=1))  # generates 'A2'\n  tf.random.set_seed(1234)\n  print(tf.random.uniform([1], seed=1))  # generates 'A1'\n  print(tf.random.uniform([1], seed=1))  # generates 'A2'\n  ```\n\n  When multiple identical random ops are wrapped in a `tf.function`, their\n  behaviors change because the ops no long share the same counter. For example:\n\n  ```python\n  @tf.function\n  def foo():\n    a = tf.random.uniform([1], seed=1)\n    b = tf.random.uniform([1], seed=1)\n    return a, b\n  print(foo())  # prints '(A1, A1)'\n  print(foo())  # prints '(A2, A2)'\n\n  @tf.function\n  def bar():\n    a = tf.random.uniform([1])\n    b = tf.random.uniform([1])\n    return a, b\n  print(bar())  # prints '(A1, A2)'\n  print(bar())  # prints '(A3, A4)'\n  ```\n\n  The second call of `foo` returns '(A2, A2)' instead of '(A1, A1)' because\n  `tf.random.uniform` maintains an internal counter. If you want `foo` to return\n  '(A1, A1)' every time, use the stateless random ops such as\n  `tf.random.stateless_uniform`. Also see `tf.random.experimental.Generator` for\n  a new set of stateful random ops that use external variables to manage their\n  states.\n\n  Args:\n    seed: integer.\n  "
    set_random_seed(seed)