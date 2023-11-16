"""The implementation of `tf.data.Dataset.take_while`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops

def _take_while(input_dataset, predicate, name=None):
    if False:
        print('Hello World!')
    'See `Dataset.take_while()` for details.'
    return _TakeWhileDataset(input_dataset, predicate, name=name)

class _TakeWhileDataset(dataset_ops.UnaryUnchangedStructureDataset):
    """A dataset that stops iteration when `predicate` returns false."""

    def __init__(self, input_dataset, predicate, name=None):
        if False:
            for i in range(10):
                print('nop')
        'See `take_while()` for details.'
        self._input_dataset = input_dataset
        wrapped_func = structured_function.StructuredFunctionWrapper(predicate, self._transformation_name(), dataset=self._input_dataset)
        if not wrapped_func.output_structure.is_compatible_with(tensor_spec.TensorSpec([], dtypes.bool)):
            raise ValueError(f'Invalid `predicate`. `predicate` must return a `tf.bool` scalar tensor but its return type is{wrapped_func.output_structure}.')
        self._predicate = wrapped_func
        self._name = name
        variant_tensor = ged_ops.take_while_dataset(self._input_dataset._variant_tensor, other_arguments=self._predicate.function.captured_inputs, predicate=self._predicate.function, **self._common_args)
        super().__init__(input_dataset, variant_tensor)

    def _functions(self):
        if False:
            while True:
                i = 10
        return [self._predicate]

    def _transformation_name(self):
        if False:
            while True:
                i = 10
        return 'Dataset.take_while()'