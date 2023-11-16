import math
import numpy as np
from bigdl.dllib.utils.log4Error import *

def calc_output_shape(input, kernel, padding=0, stride=1, dilation=1, ceil_mode=False):
    if False:
        i = 10
        return i + 15

    def dilated_kernel_size(kernel, dilation):
        if False:
            print('Hello World!')
        return kernel + (kernel - 1) * (dilation - 1)
    rounding = math.ceil if ceil_mode else math.floor
    out = (input + 2 * padding - dilated_kernel_size(kernel, dilation)) / stride + 1
    out = int(rounding(out))
    return out

def parse_node_attr(node_proto):
    if False:
        print('Hello World!')
    attrs = {}
    attr_proto = node_proto.attribute
    for attr in attr_proto:
        for field in ['f', 'i', 's']:
            if attr.HasField(field):
                attrs[attr.name] = getattr(attr, field)
                if isinstance(attrs[attr.name], bytes):
                    attrs[attr.name] = attrs[attr.name].decode(encoding='utf-8')
        for field in ['floats', 'ints', 'strings']:
            if list(getattr(attr, field)):
                invalidInputError(attr.name not in attrs, 'Only one type of attr is allowed')
                attrs[attr.name] = tuple(getattr(attr, field))
        for field in ['t', 'g']:
            if attr.HasField(field):
                attrs[attr.name] = getattr(attr, field)
        for field in ['tensors', 'graphs']:
            if list(getattr(attr, field)):
                invalidInputError(False, 'Not implement yet')
        if attr.name not in attrs:
            invalidInputError(False, 'Cannot parse attribute: \n{}\n.'.format(attr))
    return attrs

def parse_tensor_data(tensor_proto):
    if False:
        for i in range(10):
            print('nop')
    try:
        from onnx.numpy_helper import to_array
    except ImportError:
        invalidInputError(False, 'Onnx and protobuf need to be installed.')
    if len(tuple(tensor_proto.dims)) > 0:
        np_array = to_array(tensor_proto).reshape(tuple(tensor_proto.dims))
    else:
        np_array = np.array([to_array(tensor_proto)])
    return np_array