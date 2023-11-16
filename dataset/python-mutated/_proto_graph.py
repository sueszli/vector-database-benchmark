from typing import Optional
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto

def attr_value_proto(dtype, shape, s):
    if False:
        while True:
            i = 10
    "Create a dict of objects matching a NodeDef's attr field.\n\n    Follows https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/attr_value.proto\n    specifically designed for a NodeDef. The values have been reverse engineered from\n    standard TensorBoard logged data.\n    "
    attr = {}
    if s is not None:
        attr['attr'] = AttrValue(s=s.encode(encoding='utf_8'))
    if shape is not None:
        shapeproto = tensor_shape_proto(shape)
        attr['_output_shapes'] = AttrValue(list=AttrValue.ListValue(shape=[shapeproto]))
    return attr

def tensor_shape_proto(outputsize):
    if False:
        i = 10
        return i + 15
    'Create an object matching a tensor_shape field.\n\n    Follows https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/tensor_shape.proto .\n    '
    return TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in outputsize])

def node_proto(name, op='UnSpecified', input=None, dtype=None, shape: Optional[tuple]=None, outputsize=None, attributes=''):
    if False:
        return 10
    'Create an object matching a NodeDef.\n\n    Follows https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/node_def.proto .\n    '
    if input is None:
        input = []
    if not isinstance(input, list):
        input = [input]
    return NodeDef(name=name.encode(encoding='utf_8'), op=op, input=input, attr=attr_value_proto(dtype, outputsize, attributes))