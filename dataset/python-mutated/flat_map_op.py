"""The implementation of `tf.data.Dataset.flat_map`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.ops import gen_dataset_ops

def _flat_map(input_dataset, map_func, name=None):
    if False:
        i = 10
        return i + 15
    'See `Dataset.flat_map()` for details.'
    return _FlatMapDataset(input_dataset, map_func, name)

class _FlatMapDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that maps a function over its input and flattens the result."""

    def __init__(self, input_dataset, map_func, name=None):
        if False:
            print('Hello World!')
        self._input_dataset = input_dataset
        self._map_func = structured_function.StructuredFunctionWrapper(map_func, self._transformation_name(), dataset=input_dataset)
        if not isinstance(self._map_func.output_structure, dataset_ops.DatasetSpec):
            raise TypeError(f'The `map_func` argument must return a `Dataset` object. Got {dataset_ops.get_type(self._map_func.output_structure)!r}.')
        self._structure = self._map_func.output_structure._element_spec
        self._name = name
        variant_tensor = gen_dataset_ops.flat_map_dataset(input_dataset._variant_tensor, self._map_func.function.captured_inputs, f=self._map_func.function, **self._common_args)
        super().__init__(input_dataset, variant_tensor)

    def _functions(self):
        if False:
            while True:
                i = 10
        return [self._map_func]

    @property
    def element_spec(self):
        if False:
            i = 10
            return i + 15
        return self._structure

    def _transformation_name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Dataset.flat_map()'