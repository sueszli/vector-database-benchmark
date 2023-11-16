"""The implementation of `tf.data.Dataset.group_by_window`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops

def _group_by_window(input_dataset, key_func, reduce_func, window_size=None, window_size_func=None, name=None):
    if False:
        while True:
            i = 10
    'See `Dataset.group_by_window()` for details.'
    if window_size is not None and window_size_func or not (window_size is not None or window_size_func):
        raise ValueError('Either the `window_size` argument or the `window_size_func` argument must be specified.')
    if window_size is not None:

        def constant_window_func(unused_key):
            if False:
                for i in range(10):
                    print('nop')
            return ops.convert_to_tensor(window_size, dtype=dtypes.int64)
        window_size_func = constant_window_func
    assert window_size_func is not None
    return _GroupByWindowDataset(input_dataset, key_func, reduce_func, window_size_func, name=name)

class _GroupByWindowDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that groups its input and performs a windowed reduction."""

    def __init__(self, input_dataset, key_func, reduce_func, window_size_func, name=None):
        if False:
            return 10
        'See `group_by_window()` for details.'
        self._input_dataset = input_dataset
        self._make_key_func(key_func, input_dataset)
        self._make_reduce_func(reduce_func, input_dataset)
        self._make_window_size_func(window_size_func)
        self._name = name
        variant_tensor = ged_ops.group_by_window_dataset(self._input_dataset._variant_tensor, self._key_func.function.captured_inputs, self._reduce_func.function.captured_inputs, self._window_size_func.function.captured_inputs, key_func=self._key_func.function, reduce_func=self._reduce_func.function, window_size_func=self._window_size_func.function, **self._common_args)
        super().__init__(input_dataset, variant_tensor)

    def _make_window_size_func(self, window_size_func):
        if False:
            return 10
        'Make wrapping defun for window_size_func.'

        def window_size_func_wrapper(key):
            if False:
                print('Hello World!')
            return ops.convert_to_tensor(window_size_func(key), dtype=dtypes.int64)
        self._window_size_func = structured_function.StructuredFunctionWrapper(window_size_func_wrapper, self._transformation_name(), input_structure=tensor_spec.TensorSpec([], dtypes.int64))
        if not self._window_size_func.output_structure.is_compatible_with(tensor_spec.TensorSpec([], dtypes.int64)):
            raise ValueError(f'Invalid `window_size_func`. `window_size_func` must return a single `tf.int64` scalar tensor but its return type is {self._window_size_func.output_structure}.')

    def _make_key_func(self, key_func, input_dataset):
        if False:
            i = 10
            return i + 15
        'Make wrapping defun for key_func.'

        def key_func_wrapper(*args):
            if False:
                for i in range(10):
                    print('nop')
            return ops.convert_to_tensor(key_func(*args), dtype=dtypes.int64)
        self._key_func = structured_function.StructuredFunctionWrapper(key_func_wrapper, self._transformation_name(), dataset=input_dataset)
        if not self._key_func.output_structure.is_compatible_with(tensor_spec.TensorSpec([], dtypes.int64)):
            raise ValueError(f'Invalid `key_func`. `key_func` must return a single `tf.int64` scalar tensor but its return type is {self._key_func.output_structure}.')

    def _make_reduce_func(self, reduce_func, input_dataset):
        if False:
            while True:
                i = 10
        'Make wrapping defun for reduce_func.'
        nested_dataset = dataset_ops.DatasetSpec(input_dataset.element_spec)
        input_structure = (tensor_spec.TensorSpec([], dtypes.int64), nested_dataset)
        self._reduce_func = structured_function.StructuredFunctionWrapper(reduce_func, self._transformation_name(), input_structure=input_structure)
        if not isinstance(self._reduce_func.output_structure, dataset_ops.DatasetSpec):
            raise TypeError(f'Invalid `reduce_func`. `reduce_func` must return a single `tf.data.Dataset` object but its return type is {self._reduce_func.output_structure}.')
        self._element_spec = self._reduce_func.output_structure._element_spec

    @property
    def element_spec(self):
        if False:
            return 10
        return self._element_spec

    def _functions(self):
        if False:
            while True:
                i = 10
        return [self._key_func, self._reduce_func, self._window_size_func]

    def _transformation_name(self):
        if False:
            i = 10
            return i + 15
        return 'Dataset.group_by_window()'