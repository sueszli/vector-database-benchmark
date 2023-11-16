"""Python debug mode enabler."""
from tensorflow.python.eager import context
from tensorflow.python.util.tf_export import tf_export
DEBUG_MODE = False

@tf_export('data.experimental.enable_debug_mode')
def enable_debug_mode():
    if False:
        while True:
            i = 10
    'Enables debug mode for tf.data.\n\n  Example usage with pdb module:\n  ```\n  import tensorflow as tf\n  import pdb\n\n  tf.data.experimental.enable_debug_mode()\n\n  def func(x):\n    # Python 3.7 and older requires `pdb.Pdb(nosigint=True).set_trace()`\n    pdb.set_trace()\n    x = x + 1\n    return x\n\n  dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])\n  dataset = dataset.map(func)\n\n  for item in dataset:\n    print(item)\n  ```\n\n  The effect of debug mode is two-fold:\n\n  1) Any transformations that would introduce asynchrony, parallelism, or\n  non-determinism to the input pipeline execution will be forced to execute\n  synchronously, sequentially, and deterministically.\n\n  2) Any user-defined functions passed into tf.data transformations such as\n  `map` will be wrapped in `tf.py_function` so that their body is executed\n  "eagerly" as a Python function as opposed to a traced TensorFlow graph, which\n  is the default behavior. Note that even when debug mode is enabled, the\n  user-defined function is still traced  to infer the shape and type of its\n  outputs; as a consequence, any `print` statements or breakpoints will be\n  triggered once during the tracing before the actual execution of the input\n  pipeline.\n\n  NOTE: As the debug mode setting affects the construction of the tf.data input\n  pipeline, it should be enabled before any tf.data definitions.\n\n  Raises:\n    ValueError: When invoked from graph mode.\n  '
    if context.executing_eagerly():
        toggle_debug_mode(True)
    else:
        raise ValueError('`enable_debug_mode() is only supported in eager mode.')

def toggle_debug_mode(debug_mode):
    if False:
        return 10
    global DEBUG_MODE
    DEBUG_MODE = debug_mode