"""Resampling dataset transformations."""
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

@deprecation.deprecated(None, 'Use `tf.data.Dataset.rejection_resample(...)`.')
@tf_export('data.experimental.rejection_resample')
def rejection_resample(class_func, target_dist, initial_dist=None, seed=None):
    if False:
        i = 10
        return i + 15
    'A transformation that resamples a dataset to achieve a target distribution.\n\n  **NOTE** Resampling is performed via rejection sampling; some fraction\n  of the input values will be dropped.\n\n  Args:\n    class_func: A function mapping an element of the input dataset to a scalar\n      `tf.int32` tensor. Values should be in `[0, num_classes)`.\n    target_dist: A floating point type tensor, shaped `[num_classes]`.\n    initial_dist: (Optional.)  A floating point type tensor, shaped\n      `[num_classes]`.  If not provided, the true class distribution is\n      estimated live in a streaming fashion.\n    seed: (Optional.) Python integer seed for the resampler.\n\n  Returns:\n    A `Dataset` transformation function, which can be passed to\n    `tf.data.Dataset.apply`.\n  '

    def _apply_fn(dataset):
        if False:
            print('Hello World!')
        'Function from `Dataset` to `Dataset` that applies the transformation.'
        return dataset.rejection_resample(class_func=class_func, target_dist=target_dist, initial_dist=initial_dist, seed=seed)
    return _apply_fn