"""TensorFlow Hub internal utilities to handle information about tensors.

This file provides utilities to refer to properties of un-instantiated Tensors
in a concise way. Note: Ideally TensorFlow would provide a way to do this.
"""
import tensorflow as tf
from tensorflow_hub import tf_utils

class ParsedTensorInfo(object):
    """This is a tensor-looking object with information about a Tensor.

  This class provides a subset of methods and attributes provided by real
  instantiated Tensor/SparseTensors/CompositeTensors in a graph such that code
  designed to handle instances of it would mostly work in real Tensors.
  """

    def __init__(self, dtype, shape, is_sparse, type_spec=None):
        if False:
            i = 10
            return i + 15
        if type_spec is not None:
            assert dtype is None and shape is None and (is_sparse is None)
            self._type_spec = type_spec
        elif is_sparse:
            self._type_spec = tf.SparseTensorSpec(shape, dtype)
        else:
            self._type_spec = tf.TensorSpec(shape, dtype)

    @classmethod
    def from_type_spec(cls, type_spec):
        if False:
            while True:
                i = 10
        return cls(None, None, None, type_spec)

    @property
    def dtype(self):
        if False:
            i = 10
            return i + 15
        'The `DType` of elements in this tensor.'
        if hasattr(self._type_spec, 'dtype'):
            return self._type_spec.dtype
        elif hasattr(self._type_spec, '_dtype'):
            return self._type_spec._dtype
        else:
            raise ValueError('Expected TypeSpec %r to have a dtype attribute' % self._type_spec)

    def get_shape(self):
        if False:
            i = 10
            return i + 15
        'The `TensorShape` that represents the dense shape of this tensor.'
        if hasattr(self._type_spec, 'shape'):
            return self._type_spec.shape
        elif hasattr(self._type_spec, '_shape'):
            return self._type_spec._shape
        else:
            raise ValueError('Expected TypeSpec %r to have a shape attribute' % self._type_spec)

    @property
    def is_sparse(self):
        if False:
            i = 10
            return i + 15
        'Whether it represents a sparse tensor.'
        return isinstance(self._type_spec, tf.SparseTensorSpec)

    @property
    def is_composite(self):
        if False:
            while True:
                i = 10
        'Whether it represents a composite tensor.  (True for SparseTensor.)'
        return not isinstance(self._type_spec, tf.TensorSpec)

    @property
    def type_spec(self):
        if False:
            i = 10
            return i + 15
        "`tf.TypeSpec` describing this value's type."
        return self._type_spec

    @property
    def is_supported_type(self):
        if False:
            for i in range(10):
                print('nop')
        return issubclass(self._type_spec.value_type, tf_utils.SUPPORTED_ARGUMENT_TYPES)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self._type_spec, (tf.TensorSpec, tf.SparseTensorSpec)):
            return '<hub.ParsedTensorInfo shape=%s dtype=%s is_sparse=%s>' % (self.get_shape(), self.dtype.name, self.is_sparse)
        else:
            return '<hub.ParsedTensorInfo type_spec=%s>' % self.type_spec

def _parse_tensor_info_proto(tensor_info):
    if False:
        return 10
    'Returns a ParsedTensorInfo instance from a TensorInfo proto.'
    encoding = tensor_info.WhichOneof('encoding')
    if encoding == 'name':
        dtype = tf.DType(tensor_info.dtype)
        shape = tf.TensorShape(tensor_info.tensor_shape)
        return ParsedTensorInfo(dtype=dtype, shape=shape, is_sparse=False)
    elif encoding == 'coo_sparse':
        dtype = tf.DType(tensor_info.dtype)
        shape = tf.TensorShape(tensor_info.tensor_shape)
        return ParsedTensorInfo(dtype=dtype, shape=shape, is_sparse=True)
    elif encoding == 'composite_tensor':
        spec = tf_utils.composite_tensor_info_to_type_spec(tensor_info)
        return ParsedTensorInfo.from_type_spec(spec)
    else:
        raise ValueError('Unsupported TensorInfo encoding %r' % encoding)

def parse_tensor_info_map(protomap):
    if False:
        print('Hello World!')
    'Converts a proto map<string, TensorInfo> into a native Python dict.\n\n  The keys are preserved. The TensorInfo protos are parsed into objects\n  with dtype property and get_shape() method similar to Tensor, SparseTensor,\n  and RaggedTensor objects, and additional `is_sparse` and `is_composite`\n  properties.\n\n  Args:\n    protomap: A proto map<string, TensorInfo>.\n\n  Returns:\n    A map from the original keys to python objects.\n  '
    return {key: _parse_tensor_info_proto(value) for (key, value) in protomap.items()}

def _get_type_spec(value):
    if False:
        return 10
    if isinstance(value, ParsedTensorInfo):
        return value.type_spec
    elif tf_utils.is_composite_tensor(value):
        return tf_utils.get_composite_tensor_type_spec(value)
    else:
        return tf.TensorSpec.from_tensor(value)

def _convert_to_compatible_tensor(value, target, error_prefix):
    if False:
        print('Hello World!')
    'Converts `value` into a tensor that can be feed into `tensor_info`.\n\n  Args:\n    value: A value to convert into Tensor or CompositeTensor.\n    target: An object returned by `parse_tensor_info_map`.\n    error_prefix: A string to prefix on raised TypeErrors.\n\n  Raises:\n    TypeError: If it fails to convert.\n\n  Returns:\n    A Tensor or CompositeTensor compatible with tensor_info.\n  '
    if tf_utils.is_composite_tensor(value):
        tensor = value
    else:
        try:
            tensor = tf.compat.v1.convert_to_tensor_or_indexed_slices(value, target.dtype)
        except TypeError as e:
            raise TypeError('%s: %s' % (error_prefix, e))
    tensor_type_spec = _get_type_spec(tensor)
    target_type_spec = _get_type_spec(target)
    if not ParsedTensorInfo.from_type_spec(tensor_type_spec).is_supported_type:
        raise ValueError('%s: Passed argument of type %s, which is not supported by this version of tensorflow_hub.' % (error_prefix, tensor_type_spec.value_type.__name__))
    if not tensor_type_spec.is_compatible_with(target_type_spec):
        if tensor_type_spec.value_type != target_type_spec.value_type:
            got = tensor_type_spec.value_type.__name__
            expected = target_type_spec.value_type.__name__
        else:
            got = str(tensor_type_spec)
            expected = str(target_type_spec)
        raise TypeError('%s: Got %s. Expected %s.' % (error_prefix, got, expected))
    return tensor

def convert_dict_to_compatible_tensor(values, targets):
    if False:
        while True:
            i = 10
    'Converts dict `values` in tensors that are compatible with `targets`.\n\n  Args:\n    values: A dict to objects to convert with same keys as `targets`.\n    targets: A dict returned by `parse_tensor_info_map`.\n\n  Returns:\n    A map with the same keys as `values` but values converted into\n    Tensor/CompositeTensor that can be fed into `protomap`.\n\n  Raises:\n    TypeError: If it fails to convert.\n  '
    result = {}
    for (key, value) in sorted(values.items()):
        result[key] = _convert_to_compatible_tensor(value, targets[key], error_prefix="Can't convert %r" % key)
    return result

def build_input_map(protomap, inputs):
    if False:
        print('Hello World!')
    'Builds a map to feed tensors in `protomap` using `inputs`.\n\n  Args:\n    protomap: A proto map<string,TensorInfo>.\n    inputs: A map with same keys as `protomap` of Tensors and CompositeTensors.\n\n  Returns:\n    A map from nodes refered by TensorInfo protos to corresponding input\n    tensors.\n\n  Raises:\n    ValueError: if a TensorInfo proto is malformed or map keys do not match.\n  '
    if set(protomap.keys()) != set(inputs.keys()):
        raise ValueError('build_input_map: keys do not match.')
    input_map = {}
    for (key, tensor_info) in protomap.items():
        arg = inputs[key]
        encoding = tensor_info.WhichOneof('encoding')
        if encoding == 'name':
            input_map[tensor_info.name] = arg
        elif encoding == 'coo_sparse':
            coo_sparse = tensor_info.coo_sparse
            input_map[coo_sparse.values_tensor_name] = arg.values
            input_map[coo_sparse.indices_tensor_name] = arg.indices
            input_map[coo_sparse.dense_shape_tensor_name] = arg.dense_shape
        elif encoding == 'composite_tensor':
            component_infos = tensor_info.composite_tensor.components
            component_tensors = tf.nest.flatten(arg, expand_composites=True)
            for (info, tensor) in zip(component_infos, component_tensors):
                input_map[info.name] = tensor
        else:
            raise ValueError('Invalid TensorInfo.encoding: %s' % encoding)
    return input_map

def build_output_map(protomap, get_tensor_by_name):
    if False:
        i = 10
        return i + 15
    'Builds a map of tensors from `protomap` using `get_tensor_by_name`.\n\n  Args:\n    protomap: A proto map<string,TensorInfo>.\n    get_tensor_by_name: A lambda that receives a tensor name and returns a\n      Tensor instance.\n\n  Returns:\n    A map from string to Tensor or CompositeTensor instances built from\n    `protomap` and resolving tensors using `get_tensor_by_name()`.\n\n  Raises:\n    ValueError: if a TensorInfo proto is malformed.\n  '

    def get_output_from_tensor_info(tensor_info):
        if False:
            i = 10
            return i + 15
        encoding = tensor_info.WhichOneof('encoding')
        if encoding == 'name':
            return get_tensor_by_name(tensor_info.name)
        elif encoding == 'coo_sparse':
            return tf.SparseTensor(get_tensor_by_name(tensor_info.coo_sparse.indices_tensor_name), get_tensor_by_name(tensor_info.coo_sparse.values_tensor_name), get_tensor_by_name(tensor_info.coo_sparse.dense_shape_tensor_name))
        elif encoding == 'composite_tensor':
            type_spec = tf_utils.composite_tensor_info_to_type_spec(tensor_info)
            components = [get_tensor_by_name(component.name) for component in tensor_info.composite_tensor.components]
            return tf_utils.composite_tensor_from_components(type_spec, components)
        else:
            raise ValueError('Invalid TensorInfo.encoding: %s' % encoding)
    return {key: get_output_from_tensor_info(tensor_info) for (key, tensor_info) in protomap.items()}

def tensor_info_proto_maps_match(map_a, map_b):
    if False:
        while True:
            i = 10
    'Whether two signature inputs/outputs match in dtype, shape and sparsity.\n\n  Args:\n    map_a: A proto map<string,TensorInfo>.\n    map_b: A proto map<string,TensorInfo>.\n\n  Returns:\n    A boolean whether `map_a` and `map_b` tensors have the same dtype, shape and\n    sparsity.\n  '
    iter_a = sorted(parse_tensor_info_map(map_a).items())
    iter_b = sorted(parse_tensor_info_map(map_b).items())
    if len(iter_a) != len(iter_b):
        return False
    for (info_a, info_b) in zip(iter_a, iter_b):
        if info_a[0] != info_b[0]:
            return False
        if info_a[1].type_spec != info_b[1].type_spec:
            return False
    return True