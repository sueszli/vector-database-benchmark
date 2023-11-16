"""Eager semantics for polymorphic function."""
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
RUN_FUNCTIONS_EAGERLY = False

@tf_export('config.functions_run_eagerly')
def functions_run_eagerly():
    if False:
        return 10
    'Returns the value of the `run_functions_eagerly` setting.'
    return RUN_FUNCTIONS_EAGERLY

@tf_export('config.run_functions_eagerly')
def run_functions_eagerly(run_eagerly):
    if False:
        print('Hello World!')
    'Enables / disables eager execution of `tf.function`s.\n\n  Calling `tf.config.run_functions_eagerly(True)` will make all\n  invocations of `tf.function` run eagerly instead of running as a traced graph\n  function. This can be useful for debugging. As the code now runs line-by-line,\n  you can add arbitrary `print` messages or pdb breakpoints to monitor the\n  inputs/outputs of each Tensorflow operation. However, you should avoid using\n  this for actual production because it significantly slows down execution.\n\n  >>> def my_func(a):\n  ...  print(f\'a: {a}\')\n  ...  return a + a\n  >>> a_fn = tf.function(my_func)\n\n  >>> # A side effect the first time the function is traced\n  >>> # In tracing time, `a` is printed with shape and dtype only\n  >>> a_fn(tf.constant(1))\n  a: Tensor("a:0", shape=(), dtype=int32)\n  <tf.Tensor: shape=(), dtype=int32, numpy=2>\n\n  >>> # `print` is a python side effect, it won\'t execute as the traced function\n  >>> # is called\n  >>> a_fn(tf.constant(2))\n  <tf.Tensor: shape=(), dtype=int32, numpy=4>\n\n  >>> # Now, switch to eager running\n  >>> tf.config.run_functions_eagerly(True)\n  >>> # The code now runs eagerly and the actual value of `a` is printed\n  >>> a_fn(tf.constant(2))\n  a: 2\n  <tf.Tensor: shape=(), dtype=int32, numpy=4>\n\n  >>> # Turn this back off\n  >>> tf.config.run_functions_eagerly(False)\n\n  Note: This flag has no effect on functions passed into tf.data transformations\n  as arguments. tf.data functions are never executed eagerly and are always\n  executed as a compiled Tensorflow Graph.\n\n  Args:\n    run_eagerly: Boolean. Whether to run functions eagerly.\n  '
    global RUN_FUNCTIONS_EAGERLY
    RUN_FUNCTIONS_EAGERLY = bool(run_eagerly)

@deprecation.deprecated(None, 'Use `tf.config.run_functions_eagerly` instead of the experimental version.')
@tf_export('config.experimental_run_functions_eagerly')
def experimental_run_functions_eagerly(run_eagerly):
    if False:
        i = 10
        return i + 15
    'Enables / disables eager execution of `tf.function`s.\n\n  Calling `tf.config.experimental_run_functions_eagerly(True)` will make all\n  invocations of `tf.function` run eagerly instead of running as a traced graph\n  function.\n\n  See `tf.config.run_functions_eagerly` for an example.\n\n  Note: This flag has no effect on functions passed into tf.data transformations\n  as arguments. tf.data functions are never executed eagerly and are always\n  executed as a compiled Tensorflow Graph.\n\n  Args:\n    run_eagerly: Boolean. Whether to run functions eagerly.\n\n  Returns:\n    None\n  '
    return run_functions_eagerly(run_eagerly)

@deprecation.deprecated(None, 'Use tf.config.functions_run_eagerly instead of the experimental version.')
@tf_export('config.experimental_functions_run_eagerly')
def experimental_functions_run_eagerly():
    if False:
        for i in range(10):
            print('nop')
    'Returns the value of the `experimental_run_functions_eagerly` setting.'
    return functions_run_eagerly()