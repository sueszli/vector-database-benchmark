"""Python API for creating a dataset from a list."""
import itertools
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.util.tf_export import tf_export

class _ListDataset(dataset_ops.DatasetSource):
    """A `Dataset` of elements from a list."""

    def __init__(self, elements, name=None):
        if False:
            while True:
                i = 10
        if not elements:
            raise ValueError('Invalid `elements`. `elements` should not be empty.')
        if not isinstance(elements, list):
            raise ValueError('Invalid `elements`. `elements` must be a list.')
        elements = [structure.normalize_element(element) for element in elements]
        type_specs = [structure.type_spec_from_value(element) for element in elements]
        num_elements = len(elements)
        for i in range(1, num_elements):
            nest.assert_same_structure(type_specs[0], type_specs[i])
        flattened_type_specs = [nest.flatten(type_spec) for type_spec in type_specs]
        num_tensors_per_element = len(flattened_type_specs[0])
        flattened_structure = [None] * num_tensors_per_element
        for i in range(num_tensors_per_element):
            flattened_structure[i] = flattened_type_specs[0][i]
            for j in range(1, num_elements):
                flattened_structure[i] = flattened_structure[i].most_specific_common_supertype([flattened_type_specs[j][i]])
        if not isinstance(type_specs[0], dataset_ops.DatasetSpec):
            self._tensors = list(itertools.chain.from_iterable([nest.flatten(element) for element in elements]))
        else:
            self._tensors = [x._variant_tensor for x in elements]
        self._structure = nest.pack_sequence_as(type_specs[0], flattened_structure)
        self._name = name
        variant_tensor = gen_experimental_dataset_ops.list_dataset(self._tensors, output_types=self._flat_types, output_shapes=self._flat_shapes, metadata=self._metadata.SerializeToString())
        super(_ListDataset, self).__init__(variant_tensor)

    @property
    def element_spec(self):
        if False:
            i = 10
            return i + 15
        return self._structure

@tf_export('data.experimental.from_list')
def from_list(elements, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Creates a `Dataset` comprising the given list of elements.\n\n  The returned dataset will produce the items in the list one by one. The\n  functionality is identical to `Dataset.from_tensor_slices` when elements are\n  scalars, but different when elements have structure. Consider the following\n  example.\n\n  >>> dataset = tf.data.experimental.from_list([(1, 'a'), (2, 'b'), (3, 'c')])\n  >>> list(dataset.as_numpy_iterator())\n  [(1, b'a'), (2, b'b'), (3, b'c')]\n\n  To get the same output with `from_tensor_slices`, the data needs to be\n  reorganized:\n\n  >>> dataset = tf.data.Dataset.from_tensor_slices(([1, 2, 3], ['a', 'b', 'c']))\n  >>> list(dataset.as_numpy_iterator())\n  [(1, b'a'), (2, b'b'), (3, b'c')]\n\n  Unlike `from_tensor_slices`, `from_list` supports non-rectangular input:\n\n  >>> dataset = tf.data.experimental.from_list([[1], [2, 3]])\n  >>> list(dataset.as_numpy_iterator())\n  [array([1], dtype=int32), array([2, 3], dtype=int32)]\n\n  Achieving the same with `from_tensor_slices` requires the use of ragged\n  tensors.\n\n  `from_list` can be more performant than `from_tensor_slices` in some cases,\n  since it avoids the need for data slicing each epoch. However, it can also be\n  less performant, because data is stored as many small tensors rather than a\n  few large tensors as in `from_tensor_slices`. The general guidance is to\n  prefer `from_list` from a performance perspective when the number of elements\n  is small (less than 1000).\n\n  Args:\n    elements: A list of elements whose components have the same nested\n      structure.\n    name: (Optional.) A name for the tf.data operation.\n\n  Returns:\n    Dataset: A `Dataset` of the `elements`.\n  "
    return _ListDataset(elements, name)