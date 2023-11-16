"""Enumerate dataset transformations."""
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

@deprecation.deprecated(None, 'Use `tf.data.Dataset.enumerate()`.')
@tf_export('data.experimental.enumerate_dataset')
def enumerate_dataset(start=0):
    if False:
        print('Hello World!')
    "A transformation that enumerates the elements of a dataset.\n\n  It is similar to python's `enumerate`.\n  For example:\n\n  ```python\n  # NOTE: The following examples use `{ ... }` to represent the\n  # contents of a dataset.\n  a = { 1, 2, 3 }\n  b = { (7, 8), (9, 10) }\n\n  # The nested structure of the `datasets` argument determines the\n  # structure of elements in the resulting dataset.\n  a.apply(tf.data.experimental.enumerate_dataset(start=5))\n  => { (5, 1), (6, 2), (7, 3) }\n  b.apply(tf.data.experimental.enumerate_dataset())\n  => { (0, (7, 8)), (1, (9, 10)) }\n  ```\n\n  Args:\n    start: A `tf.int64` scalar `tf.Tensor`, representing the start value for\n      enumeration.\n\n  Returns:\n    A `Dataset` transformation function, which can be passed to\n    `tf.data.Dataset.apply`.\n  "

    def _apply_fn(dataset):
        if False:
            i = 10
            return i + 15
        return dataset.enumerate(start)
    return _apply_fn