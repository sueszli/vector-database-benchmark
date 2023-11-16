"""Cardinality analysis of `Dataset` objects."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util.tf_export import tf_export
INFINITE = -1
UNKNOWN = -2
tf_export('data.experimental.INFINITE_CARDINALITY').export_constant(__name__, 'INFINITE')
tf_export('data.experimental.UNKNOWN_CARDINALITY').export_constant(__name__, 'UNKNOWN')

@tf_export('data.experimental.cardinality')
def cardinality(dataset):
    if False:
        i = 10
        return i + 15
    'Returns the cardinality of `dataset`, if known.\n\n  The operation returns the cardinality of `dataset`. The operation may return\n  `tf.data.experimental.INFINITE_CARDINALITY` if `dataset` contains an infinite\n  number of elements or `tf.data.experimental.UNKNOWN_CARDINALITY` if the\n  analysis fails to determine the number of elements in `dataset` (e.g. when the\n  dataset source is a file).\n\n  >>> dataset = tf.data.Dataset.range(42)\n  >>> print(tf.data.experimental.cardinality(dataset).numpy())\n  42\n  >>> dataset = dataset.repeat()\n  >>> cardinality = tf.data.experimental.cardinality(dataset)\n  >>> print((cardinality == tf.data.experimental.INFINITE_CARDINALITY).numpy())\n  True\n  >>> dataset = dataset.filter(lambda x: True)\n  >>> cardinality = tf.data.experimental.cardinality(dataset)\n  >>> print((cardinality == tf.data.experimental.UNKNOWN_CARDINALITY).numpy())\n  True\n\n  Args:\n    dataset: A `tf.data.Dataset` for which to determine cardinality.\n\n  Returns:\n    A scalar `tf.int64` `Tensor` representing the cardinality of `dataset`. If\n    the cardinality is infinite or unknown, the operation returns the named\n    constant `INFINITE_CARDINALITY` and `UNKNOWN_CARDINALITY` respectively.\n  '
    return gen_dataset_ops.dataset_cardinality(dataset._variant_tensor)

@tf_export('data.experimental.assert_cardinality')
def assert_cardinality(expected_cardinality):
    if False:
        while True:
            i = 10
    'Asserts the cardinality of the input dataset.\n\n  NOTE: The following assumes that "examples.tfrecord" contains 42 records.\n\n  >>> dataset = tf.data.TFRecordDataset("examples.tfrecord")\n  >>> cardinality = tf.data.experimental.cardinality(dataset)\n  >>> print((cardinality == tf.data.experimental.UNKNOWN_CARDINALITY).numpy())\n  True\n  >>> dataset = dataset.apply(tf.data.experimental.assert_cardinality(42))\n  >>> print(tf.data.experimental.cardinality(dataset).numpy())\n  42\n\n  Args:\n    expected_cardinality: The expected cardinality of the input dataset.\n\n  Returns:\n    A `Dataset` transformation function, which can be passed to\n    `tf.data.Dataset.apply`.\n\n  Raises:\n    FailedPreconditionError: The assertion is checked at runtime (when iterating\n      the dataset) and an error is raised if the actual and expected cardinality\n      differ.\n  '

    def _apply_fn(dataset):
        if False:
            i = 10
            return i + 15
        return _AssertCardinalityDataset(dataset, expected_cardinality)
    return _apply_fn

class _AssertCardinalityDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A `Dataset` that assert the cardinality of its input."""

    def __init__(self, input_dataset, expected_cardinality):
        if False:
            while True:
                i = 10
        self._input_dataset = input_dataset
        self._expected_cardinality = ops.convert_to_tensor(expected_cardinality, dtype=dtypes.int64, name='expected_cardinality')
        variant_tensor = ged_ops.assert_cardinality_dataset(self._input_dataset._variant_tensor, self._expected_cardinality, **self._flat_structure)
        super(_AssertCardinalityDataset, self).__init__(input_dataset, variant_tensor)