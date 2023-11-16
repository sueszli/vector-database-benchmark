"""The implementation of `tf.data.Dataset.concatenate`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.util import nest as tf_nest

def _concatenate(input_dataset, dataset_to_concatenate, name):
    if False:
        return 10
    return _ConcatenateDataset(input_dataset, dataset_to_concatenate, name)

class _ConcatenateDataset(dataset_ops.DatasetV2):
    """A `Dataset` that concatenates its input with given dataset."""

    def __init__(self, input_dataset, dataset_to_concatenate, name=None):
        if False:
            return 10
        'See `Dataset.concatenate()` for details.'
        self._input_dataset = input_dataset
        self._dataset_to_concatenate = dataset_to_concatenate

        def common_supertype(a, b):
            if False:
                print('Hello World!')
            result = a.most_specific_common_supertype([b])
            if result is None:
                raise TypeError(f'No common supertype of {a} and {b}.')
            return result
        try:
            self._structure = tf_nest.map_structure(common_supertype, input_dataset.element_spec, dataset_to_concatenate.element_spec)
        except (TypeError, ValueError) as e:
            raise TypeError(f'Incompatible dataset elements:\n  {input_dataset.element_spec} vs.   {dataset_to_concatenate.element_spec}') from e
        self._input_datasets = [input_dataset, dataset_to_concatenate]
        self._name = name
        variant_tensor = gen_dataset_ops.concatenate_dataset(input_dataset._variant_tensor, dataset_to_concatenate._variant_tensor, **self._common_args)
        super().__init__(variant_tensor)

    def _inputs(self):
        if False:
            print('Hello World!')
        return self._input_datasets

    @property
    def element_spec(self):
        if False:
            i = 10
            return i + 15
        return self._structure