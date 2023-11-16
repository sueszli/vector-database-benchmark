"""Graph-only versions of a few op functions, for internal use only."""
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

def graph_placeholder(dtype, shape, name=None):
    if False:
        print('Hello World!')
    'Graph-only version of tf.compat.v1.placeholder(), for internal use only.'
    dtype = dtype.base_dtype
    dtype_value = attr_value_pb2.AttrValue(type=dtype.as_datatype_enum)
    if isinstance(shape, (list, tuple)):
        shape = tensor_shape.TensorShape(shape)
    shape = attr_value_pb2.AttrValue(shape=shape.as_proto())
    g = ops.get_default_graph()
    attrs = {'dtype': dtype_value, 'shape': shape}
    op = g._create_op_internal('Placeholder', [], [dtype], input_types=[], attrs=attrs, name=name)
    (result,) = op.outputs
    if op_callbacks.should_invoke_op_callbacks():
        callback_outputs = op_callbacks.invoke_op_callbacks('Placeholder', tuple(), attrs, tuple(op.outputs), op_name=name, graph=g)
        if callback_outputs is not None:
            (result,) = callback_outputs
    return result