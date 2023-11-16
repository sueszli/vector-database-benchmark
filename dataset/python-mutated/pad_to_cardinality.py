"""The implementation of `tf.data.experimental.pad_to_cardinality`."""
from collections.abc import Mapping
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

@tf_export('data.experimental.pad_to_cardinality')
def pad_to_cardinality(cardinality, mask_key='valid'):
    if False:
        print('Hello World!')
    "Pads a dataset with fake elements to reach the desired cardinality.\n\n  The dataset to pad must have a known and finite cardinality and contain\n  dictionary elements. The `mask_key` will be added to differentiate between\n  real and padding elements -- real elements will have a `<mask_key>=True` entry\n  while padding elements will have a `<mask_key>=False` entry.\n\n  Example usage:\n\n  >>> ds = tf.data.Dataset.from_tensor_slices({'a': [1, 2]})\n  >>> ds = ds.apply(tf.data.experimental.pad_to_cardinality(3))\n  >>> list(ds.as_numpy_iterator())\n  [{'a': 1, 'valid': True}, {'a': 2, 'valid': True}, {'a': 0, 'valid': False}]\n\n  This can be useful, e.g. during eval, when partial batches are undesirable but\n  it is also important not to drop any data.\n\n  ```\n  ds = ...\n  # Round up to the next full batch.\n  target_cardinality = -(-ds.cardinality() // batch_size) * batch_size\n  ds = ds.apply(tf.data.experimental.pad_to_cardinality(target_cardinality))\n  # Set `drop_remainder` so that batch shape will be known statically. No data\n  # will actually be dropped since the batch size divides the cardinality.\n  ds = ds.batch(batch_size, drop_remainder=True)\n  ```\n\n  Args:\n    cardinality: The cardinality to pad the dataset to.\n    mask_key: The key to use for identifying real vs padding elements.\n\n  Returns:\n    A dataset transformation that can be applied via `Dataset.apply()`.\n  "

    def make_filler_dataset(ds):
        if False:
            i = 10
            return i + 15
        padding = cardinality - ds.cardinality()
        filler_element = nest.map_structure(lambda spec: array_ops.zeros(spec.shape, spec.dtype), ds.element_spec)
        filler_element[mask_key] = False
        filler_dataset = dataset_ops.Dataset.from_tensors(filler_element)
        filler_dataset = filler_dataset.repeat(padding)
        return filler_dataset

    def apply_valid_mask(x):
        if False:
            while True:
                i = 10
        x[mask_key] = True
        return x

    def _apply_fn(dataset):
        if False:
            for i in range(10):
                print('nop')
        if context.executing_eagerly():
            if dataset.cardinality() < 0:
                raise ValueError(f'The dataset passed into `pad_to_cardinality` must have a known cardinalty, but has cardinality {dataset.cardinality()}')
            if dataset.cardinality() > cardinality:
                raise ValueError(f'The dataset passed into `pad_to_cardinality` must have a cardinalty less than the target cardinality ({cardinality}), but has cardinality {dataset.cardinality()}')
        if not isinstance(dataset.element_spec, Mapping):
            raise ValueError('`pad_to_cardinality` requires its input dataset to be a dictionary.')
        filler = make_filler_dataset(dataset)
        dataset = dataset.map(apply_valid_mask)
        dataset = dataset.concatenate(filler)
        return dataset
    return _apply_fn