"""The implementation of `tf.data.Dataset.from_tensors`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.ops import gen_dataset_ops

def _from_tensors(tensors, name):
    if False:
        for i in range(10):
            print('nop')
    return _TensorDataset(tensors, name)

class _TensorDataset(dataset_ops.DatasetSource):
    """A `Dataset` with a single element."""

    def __init__(self, element, name=None):
        if False:
            print('Hello World!')
        'See `tf.data.Dataset.from_tensors` for details.'
        element = structure.normalize_element(element)
        self._structure = structure.type_spec_from_value(element)
        self._tensors = structure.to_tensor_list(self._structure, element)
        self._name = name
        variant_tensor = gen_dataset_ops.tensor_dataset(self._tensors, output_shapes=structure.get_flat_tensor_shapes(self._structure), metadata=self._metadata.SerializeToString())
        super().__init__(variant_tensor)

    @property
    def element_spec(self):
        if False:
            while True:
                i = 10
        return self._structure