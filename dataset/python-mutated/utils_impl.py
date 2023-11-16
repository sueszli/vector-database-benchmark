"""SavedModel utility functions implementation."""
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
_DEPRECATION_MSG = 'This API was designed for TensorFlow v1. See https://www.tensorflow.org/guide/migrate for instructions on how to migrate your code to TensorFlow v2.'

@tf_export(v1=['saved_model.build_tensor_info', 'saved_model.utils.build_tensor_info'])
@deprecation.deprecated(None, _DEPRECATION_MSG)
def build_tensor_info(tensor):
    if False:
        i = 10
        return i + 15
    'Utility function to build TensorInfo proto from a Tensor.\n\n  Args:\n    tensor: Tensor or SparseTensor whose name, dtype and shape are used to\n        build the TensorInfo. For SparseTensors, the names of the three\n        constituent Tensors are used.\n\n  Returns:\n    A TensorInfo protocol buffer constructed based on the supplied argument.\n\n  Raises:\n    RuntimeError: If eager execution is enabled.\n\n  @compatibility(TF2)\n  This API is not compatible with eager execution as `tensor` needs to be a\n  graph tensor, and there is no replacement for it in TensorFlow 2.x. To start\n  writing programs using TensorFlow 2.x, please refer to the [Effective\n  TensorFlow 2](https://www.tensorflow.org/guide/effective_tf2) guide.\n  @end_compatibility\n  '
    if context.executing_eagerly():
        raise RuntimeError('`build_tensor_info` is not supported in eager execution.')
    return build_tensor_info_internal(tensor)

def build_tensor_info_internal(tensor):
    if False:
        print('Hello World!')
    'Utility function to build TensorInfo proto from a Tensor.'
    if isinstance(tensor, composite_tensor.CompositeTensor) and (not isinstance(tensor, sparse_tensor.SparseTensor)) and (not isinstance(tensor, resource_variable_ops.ResourceVariable)):
        return _build_composite_tensor_info_internal(tensor)
    tensor_info = meta_graph_pb2.TensorInfo(dtype=dtypes.as_dtype(tensor.dtype).as_datatype_enum, tensor_shape=tensor.get_shape().as_proto())
    if isinstance(tensor, sparse_tensor.SparseTensor):
        tensor_info.coo_sparse.values_tensor_name = tensor.values.name
        tensor_info.coo_sparse.indices_tensor_name = tensor.indices.name
        tensor_info.coo_sparse.dense_shape_tensor_name = tensor.dense_shape.name
    else:
        tensor_info.name = tensor.name
    return tensor_info

def _build_composite_tensor_info_internal(tensor):
    if False:
        return 10
    'Utility function to build TensorInfo proto from a CompositeTensor.'
    spec = tensor._type_spec
    tensor_info = meta_graph_pb2.TensorInfo()
    spec_proto = nested_structure_coder.encode_structure(spec)
    tensor_info.composite_tensor.type_spec.CopyFrom(spec_proto.type_spec_value)
    for component in nest.flatten(tensor, expand_composites=True):
        tensor_info.composite_tensor.components.add().CopyFrom(build_tensor_info_internal(component))
    return tensor_info

def build_tensor_info_from_op(op):
    if False:
        print('Hello World!')
    'Utility function to build TensorInfo proto from an Op.\n\n  Note that this function should be used with caution. It is strictly restricted\n  to TensorFlow internal use-cases only. Please make sure you do need it before\n  using it.\n\n  This utility function overloads the TensorInfo proto by setting the name to\n  the Op\'s name, dtype to DT_INVALID and tensor_shape as None. One typical usage\n  is for the Op of the call site for the defunned function:\n  ```python\n    @function.defun\n    def some_variable_initialization_fn(value_a, value_b):\n      a = value_a\n      b = value_b\n\n    value_a = constant_op.constant(1, name="a")\n    value_b = constant_op.constant(2, name="b")\n    op_info = utils.build_op_info(\n        some_variable_initialization_fn(value_a, value_b))\n  ```\n\n  Args:\n    op: An Op whose name is used to build the TensorInfo. The name that points\n        to the Op could be fetched at run time in the Loader session.\n\n  Returns:\n    A TensorInfo protocol buffer constructed based on the supplied argument.\n\n  Raises:\n    RuntimeError: If eager execution is enabled.\n  '
    if context.executing_eagerly():
        raise RuntimeError('`build_tensor_info_from_op` is not supported in eager execution.')
    return meta_graph_pb2.TensorInfo(dtype=types_pb2.DT_INVALID, tensor_shape=tensor_shape.unknown_shape().as_proto(), name=op.name)

@tf_export(v1=['saved_model.get_tensor_from_tensor_info', 'saved_model.utils.get_tensor_from_tensor_info'])
@deprecation.deprecated(None, _DEPRECATION_MSG)
def get_tensor_from_tensor_info(tensor_info, graph=None, import_scope=None):
    if False:
        while True:
            i = 10
    'Returns the Tensor or CompositeTensor described by a TensorInfo proto.\n\n  Args:\n    tensor_info: A TensorInfo proto describing a Tensor or SparseTensor or\n      CompositeTensor.\n    graph: The tf.Graph in which tensors are looked up. If None, the\n        current default graph is used.\n    import_scope: If not None, names in `tensor_info` are prefixed with this\n        string before lookup.\n\n  Returns:\n    The Tensor or SparseTensor or CompositeTensor in `graph` described by\n    `tensor_info`.\n\n  Raises:\n    KeyError: If `tensor_info` does not correspond to a tensor in `graph`.\n    ValueError: If `tensor_info` is malformed.\n  '
    graph = graph or ops.get_default_graph()

    def _get_tensor(name):
        if False:
            while True:
                i = 10
        return graph.get_tensor_by_name(ops.prepend_name_scope(name, import_scope=import_scope))
    encoding = tensor_info.WhichOneof('encoding')
    if encoding == 'name':
        return _get_tensor(tensor_info.name)
    elif encoding == 'coo_sparse':
        return sparse_tensor.SparseTensor(_get_tensor(tensor_info.coo_sparse.indices_tensor_name), _get_tensor(tensor_info.coo_sparse.values_tensor_name), _get_tensor(tensor_info.coo_sparse.dense_shape_tensor_name))
    elif encoding == 'composite_tensor':
        spec_proto = struct_pb2.StructuredValue(type_spec_value=tensor_info.composite_tensor.type_spec)
        spec = nested_structure_coder.decode_proto(spec_proto)
        components = [_get_tensor(component.name) for component in tensor_info.composite_tensor.components]
        return nest.pack_sequence_as(spec, components, expand_composites=True)
    else:
        raise ValueError(f'Invalid TensorInfo.encoding: {encoding}. Expected `coo_sparse`, `composite_tensor`, or `name` for a dense tensor.')

def get_element_from_tensor_info(tensor_info, graph=None, import_scope=None):
    if False:
        return 10
    'Returns the element in the graph described by a TensorInfo proto.\n\n  Args:\n    tensor_info: A TensorInfo proto describing an Op or Tensor by name.\n    graph: The tf.Graph in which tensors are looked up. If None, the current\n      default graph is used.\n    import_scope: If not None, names in `tensor_info` are prefixed with this\n      string before lookup.\n\n  Returns:\n    Op or tensor in `graph` described by `tensor_info`.\n\n  Raises:\n    KeyError: If `tensor_info` does not correspond to an op or tensor in `graph`\n  '
    graph = graph or ops.get_default_graph()
    return graph.as_graph_element(ops.prepend_name_scope(tensor_info.name, import_scope=import_scope))

def swap_function_tensor_content(meta_graph_def, from_endiness, to_endiness):
    if False:
        print('Hello World!')
    bst.swap_tensor_content_in_graph_function(meta_graph_def, from_endiness, to_endiness)