"""The implementation of `tf.data.Dataset.padded_batch`."""
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_dataset_ops

def _padded_batch(input_dataset, batch_size, padded_shapes=None, padding_values=None, drop_remainder=False, name=None):
    if False:
        return 10
    'See `tf.data.Dataset.padded_batch` for details.'
    if padded_shapes is None:
        padded_shapes = dataset_ops.get_legacy_output_shapes(input_dataset)
        for (i, shape) in enumerate(nest.flatten(padded_shapes)):
            if not shape:
                raise ValueError(f'You must provide `padded_shapes` argument because component {i} has unknown rank.')
    return _PaddedBatchDataset(input_dataset, batch_size, padded_shapes, padding_values, drop_remainder, name=name)

def _is_padded_shape_compatible_with(padded_shape, input_component_shape):
    if False:
        for i in range(10):
            print('nop')
    'Returns `True` if `input_component_shape` can be padded to `padded_shape`.\n\n  Args:\n    padded_shape: A `tf.TensorShape`.\n    input_component_shape: A `tf.TensorShape`.\n\n  Returns:\n    `True` if `input_component_shape` can be padded to `padded_shape`, otherwise\n    `False`.\n  '
    if padded_shape.dims is None or input_component_shape.dims is None:
        return True
    if len(padded_shape.dims) != len(input_component_shape.dims):
        return False
    for (padded_dim, input_dim) in zip(padded_shape.dims, input_component_shape.dims):
        if padded_dim.value is not None and input_dim.value is not None and (padded_dim.value < input_dim.value):
            return False
    return True

def _padded_shape_to_tensor(padded_shape, input_component_shape):
    if False:
        while True:
            i = 10
    'Converts `padded_shape` to a `tf.Tensor` representing that shape.\n\n  Args:\n    padded_shape: A shape-like object, which may be a `tf.TensorShape`, a Python\n      sequence, or a 1-D `tf.Tensor` of `tf.int64` elements.\n    input_component_shape: A `tf.TensorShape`, with which `padded_shape` must be\n      compatible.\n\n  Returns:\n    A 1-D `tf.Tensor` of `tf.int64` elements, representing `padded_shape`.\n\n  Raises:\n    ValueError: If `padded_shape` is not a shape or not compatible with\n      `input_component_shape`.\n    TypeError: If `padded_shape` is not convertible to a `tf.int64` tensor.\n  '
    try:
        padded_shape_as_shape = tensor_shape.as_shape(padded_shape)
        ret = ops.convert_to_tensor([dim if dim is not None else -1 for dim in padded_shape_as_shape.as_list()], dtype=dtypes.int64)
    except (TypeError, ValueError) as e:
        ret = ops.convert_to_tensor(padded_shape, preferred_dtype=dtypes.int64)
        if ret.shape.dims is not None and len(ret.shape.dims) != 1:
            raise ValueError(f'Padded shape {padded_shape} must be a `tf.int64` vector tensor, but its shape was {ret.shape}.') from e
        if ret.dtype != dtypes.int64:
            raise TypeError(f'Padded shape {padded_shape} must be a `tf.int64` vector tensor, but its element type was {ret.dtype.name}.') from e
        padded_shape_as_shape = tensor_util.constant_value_as_shape(ret)
    if not _is_padded_shape_compatible_with(padded_shape_as_shape, input_component_shape):
        raise ValueError(f'The padded shape {padded_shape_as_shape} is not compatible with the shape {input_component_shape} of the corresponding input component.')
    return ret

def _padding_values_or_default(padding_values, input_dataset):
    if False:
        while True:
            i = 10
    'Returns padding values with None elements replaced with default values.'

    def make_zero(t):
        if False:
            while True:
                i = 10
        if t.base_dtype == dtypes.string:
            return ''
        elif t.base_dtype == dtypes.variant:
            raise TypeError("Unable to create default padding value for a component of type 'variant'.")
        elif t.base_dtype == dtypes.bfloat16:
            return constant_op.constant(0, dtype=dtypes.bfloat16)
        else:
            return np.zeros_like(t.as_numpy_dtype())

    def value_or_default(value, default):
        if False:
            return 10
        return default if value is None else value
    default_padding = nest.map_structure(make_zero, dataset_ops.get_legacy_output_types(input_dataset))
    return nest.map_structure_up_to(padding_values, value_or_default, padding_values, default_padding)

def _padding_value_to_tensor(value, output_type):
    if False:
        for i in range(10):
            print('nop')
    "Converts the padding value to a tensor.\n\n  Args:\n    value: The padding value.\n    output_type: Its expected dtype.\n\n  Returns:\n    A scalar `Tensor`.\n\n  Raises:\n    ValueError: if the padding value is not a scalar.\n    TypeError: if the padding value's type does not match `output_type`.\n  "
    value = ops.convert_to_tensor(value, name='padding_value')
    if not value.shape.is_compatible_with(tensor_shape.TensorShape([])):
        raise ValueError(f'Invalid `padding_values`. `padding_values` values should be scalars, but got {value.shape}.')
    if value.dtype != output_type:
        raise TypeError(f'Invalid `padding_values`. `padding_values` values type {value.dtype} does not match type {output_type} of the corresponding input component.')
    return value

class _PaddedBatchDataset(dataset_ops.UnaryDataset):
    """A `Dataset` that batches and pads contiguous elements from its input."""

    def __init__(self, input_dataset, batch_size, padded_shapes, padding_values, drop_remainder, name=None):
        if False:
            i = 10
            return i + 15
        'See `Dataset.batch()` for details.'
        self._input_dataset = input_dataset

        def check_types(component_spec):
            if False:
                for i in range(10):
                    print('nop')
            if not isinstance(component_spec, tensor_spec.TensorSpec):
                if isinstance(component_spec, dataset_ops.DatasetSpec):
                    raise TypeError('`padded_batch` is not supported for datasets of datasets')
                raise TypeError(f'`padded_batch` is only supported for datasets that produce tensor elements but type spec of elements in the input dataset is not a subclass of TensorSpec: `{component_spec}`.')
        nest.map_structure(check_types, input_dataset.element_spec)
        self._input_dataset = input_dataset
        self._batch_size = ops.convert_to_tensor(batch_size, dtype=dtypes.int64, name='batch_size')
        padding_values = _padding_values_or_default(padding_values, input_dataset)
        input_shapes = dataset_ops.get_legacy_output_shapes(input_dataset)
        flat_padded_shapes = nest.flatten_up_to(input_shapes, padded_shapes)
        flat_padded_shapes_as_tensors = []
        for (input_component_shape, padded_shape) in zip(nest.flatten(input_shapes), flat_padded_shapes):
            flat_padded_shapes_as_tensors.append(_padded_shape_to_tensor(padded_shape, input_component_shape))
        self._padded_shapes = nest.pack_sequence_as(input_shapes, flat_padded_shapes_as_tensors)
        if nest.is_nested(input_shapes) and (not nest.is_nested(padding_values)):
            padding_values = nest.map_structure(lambda _: padding_values, input_shapes)
        self._padding_values = nest.map_structure_up_to(input_shapes, _padding_value_to_tensor, padding_values, dataset_ops.get_legacy_output_types(input_dataset))
        self._drop_remainder = ops.convert_to_tensor(drop_remainder, dtype=dtypes.bool, name='drop_remainder')

        def _padded_shape_to_batch_shape(s):
            if False:
                while True:
                    i = 10
            return tensor_shape.TensorShape([tensor_util.constant_value(self._batch_size) if smart_cond.smart_constant_value(self._drop_remainder) else None]).concatenate(tensor_util.constant_value_as_shape(s))
        output_shapes = nest.map_structure(_padded_shape_to_batch_shape, self._padded_shapes)
        self._structure = structure.convert_legacy_structure(dataset_ops.get_legacy_output_types(self._input_dataset), output_shapes, dataset_ops.get_legacy_output_classes(self._input_dataset))
        self._name = name
        variant_tensor = gen_dataset_ops.padded_batch_dataset_v2(input_dataset._variant_tensor, batch_size=self._batch_size, padded_shapes=[ops.convert_to_tensor(s, dtype=dtypes.int64) for s in nest.flatten(self._padded_shapes)], padding_values=nest.flatten(self._padding_values), drop_remainder=self._drop_remainder, output_shapes=structure.get_flat_tensor_shapes(self._structure), metadata=self._metadata.SerializeToString())
        super().__init__(input_dataset, variant_tensor)

    @property
    def element_spec(self):
        if False:
            print('Hello World!')
        return self._structure