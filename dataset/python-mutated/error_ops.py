"""Ignore_errors dataset transformations."""
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

@tf_export('data.experimental.ignore_errors')
@deprecation.deprecated(None, 'Use `tf.data.Dataset.ignore_errors` instead.')
def ignore_errors(log_warning=False):
    if False:
        while True:
            i = 10
    'Creates a `Dataset` from another `Dataset` and silently ignores any errors.\n\n  Use this transformation to produce a dataset that contains the same elements\n  as the input, but silently drops any elements that caused an error. For\n  example:\n\n  ```python\n  dataset = tf.data.Dataset.from_tensor_slices([1., 2., 0., 4.])\n\n  # Computing `tf.debugging.check_numerics(1. / 0.)` will raise an\n  InvalidArgumentError.\n  dataset = dataset.map(lambda x: tf.debugging.check_numerics(1. / x, "error"))\n\n  # Using `ignore_errors()` will drop the element that causes an error.\n  dataset =\n      dataset.apply(tf.data.experimental.ignore_errors())  # ==> {1., 0.5, 0.2}\n  ```\n  Args:\n     log_warning: (Optional.) A \'tf.bool\' scalar indicating whether ignored\n      errors should be logged to stderr. Defaults to \'False\'.\n\n  Returns:\n    A `Dataset` transformation function, which can be passed to\n    `tf.data.Dataset.apply`.\n  '

    def _apply_fn(dataset):
        if False:
            return 10
        return dataset.ignore_errors(log_warning)
    return _apply_fn