"""Scan dataset transformation."""
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

@deprecation.deprecated(None, 'Use `tf.data.Dataset.scan(...) instead')
@tf_export('data.experimental.scan')
def scan(initial_state, scan_func):
    if False:
        while True:
            i = 10
    'A transformation that scans a function across an input dataset.\n\n  This transformation is a stateful relative of `tf.data.Dataset.map`.\n  In addition to mapping `scan_func` across the elements of the input dataset,\n  `scan()` accumulates one or more state tensors, whose initial values are\n  `initial_state`.\n\n  Args:\n    initial_state: A nested structure of tensors, representing the initial state\n      of the accumulator.\n    scan_func: A function that maps `(old_state, input_element)` to\n      `(new_state, output_element)`. It must take two arguments and return a\n      pair of nested structures of tensors. The `new_state` must match the\n      structure of `initial_state`.\n\n  Returns:\n    A `Dataset` transformation function, which can be passed to\n    `tf.data.Dataset.apply`.\n  '

    def _apply_fn(dataset):
        if False:
            i = 10
            return i + 15
        return dataset.scan(initial_state=initial_state, scan_func=scan_func)
    return _apply_fn