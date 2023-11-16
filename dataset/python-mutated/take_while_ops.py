"""take-while dataset transformation."""
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

@deprecation.deprecated(None, 'Use `tf.data.Dataset.take_while(...)')
@tf_export('data.experimental.take_while')
def take_while(predicate):
    if False:
        for i in range(10):
            print('nop')
    'A transformation that stops dataset iteration based on a `predicate`.\n\n  Args:\n    predicate: A function that maps a nested structure of tensors (having shapes\n      and types defined by `self.output_shapes` and `self.output_types`) to a\n      scalar `tf.bool` tensor.\n\n  Returns:\n    A `Dataset` transformation function, which can be passed to\n    `tf.data.Dataset.apply`.\n  '

    def _apply_fn(dataset):
        if False:
            print('Hello World!')
        return dataset.take_while(predicate=predicate)
    return _apply_fn