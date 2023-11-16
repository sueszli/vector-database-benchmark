"""The implementation of `tf.data.Dataset.choose_from_datasets`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import directed_interleave_op
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.types import data as data_types

def _choose_from_datasets(datasets, choice_dataset, stop_on_empty_dataset=True):
    if False:
        while True:
            i = 10
    'See `Dataset.choose_from_datasets()` for details.'
    if not datasets:
        raise ValueError('Invalid `datasets`. `datasets` should not be empty.')
    if not isinstance(choice_dataset, data_types.DatasetV2):
        raise TypeError(f'Invalid `choice_dataset`. `choice_dataset` should be a `tf.data.Dataset` but is {type(choice_dataset)}.')
    if not structure.are_compatible(choice_dataset.element_spec, tensor_spec.TensorSpec([], dtypes.int64)):
        raise TypeError(f'Invalid `choice_dataset`. Elements of `choice_dataset` must be scalar `tf.int64` tensors but are {choice_dataset.element_spec}.')
    choice_dataset = dataset_ops._apply_rewrite(choice_dataset, 'replicate_on_split')
    return directed_interleave_op._directed_interleave(choice_dataset, datasets, stop_on_empty_dataset)