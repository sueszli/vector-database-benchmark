"""The implementation of `tf.data.Dataset.shuffle`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops

def _directed_interleave(selector_input, data_inputs, stop_on_empty_dataset=False):
    if False:
        i = 10
        return i + 15
    return _DirectedInterleaveDataset(selector_input, data_inputs, stop_on_empty_dataset=stop_on_empty_dataset)

class _DirectedInterleaveDataset(dataset_ops.DatasetV2):
    """A substitute for `Dataset.interleave()` on a fixed list of datasets."""

    def __init__(self, selector_input, data_inputs, stop_on_empty_dataset=False):
        if False:
            print('Hello World!')
        self._selector_input = selector_input
        self._data_inputs = list(data_inputs)
        self._stop_on_empty_dataset = stop_on_empty_dataset
        spec = self._data_inputs[0].element_spec
        for (i, data_input) in enumerate(self._data_inputs[1:]):

            def common_supertype(a, b):
                if False:
                    for i in range(10):
                        print('nop')
                result = a.most_specific_common_supertype([b])
                if result is None:
                    raise TypeError(f'No common supertype of {a} and {b}.')
                return result
            try:
                spec = nest.map_structure(common_supertype, spec, data_input.element_spec)
            except (TypeError, ValueError) as e:
                raise TypeError(f'Invalid `datasets`. `datasets` must have compatible element specs.\n Dataset 0 element_spec={data_inputs[0].element_spec}.\nDataset {i + 1} element_spec={data_input.element_spec}.') from e
        self._element_spec = spec
        variant_tensor = ged_ops.directed_interleave_dataset(self._selector_input._variant_tensor, [data_input._variant_tensor for data_input in self._data_inputs], stop_on_empty_dataset=self._stop_on_empty_dataset, **self._flat_structure)
        super().__init__(variant_tensor)

    def _inputs(self):
        if False:
            return 10
        return [self._selector_input] + self._data_inputs

    @property
    def element_spec(self):
        if False:
            while True:
                i = 10
        return self._element_spec