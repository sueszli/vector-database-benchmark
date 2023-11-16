"""Helper functions to access TensorShape values.

The rank 4 tensor_shape must be of the form [batch_size, height, width, depth].
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def get_dim_as_int(dim):
    if False:
        i = 10
        return i + 15
    'Utility to get v1 or v2 TensorShape dim as an int.\n\n  Args:\n    dim: The TensorShape dimension to get as an int\n\n  Returns:\n    None or an int.\n  '
    try:
        return dim.value
    except AttributeError:
        return dim

def get_batch_size(tensor_shape):
    if False:
        return 10
    'Returns batch size from the tensor shape.\n\n  Args:\n    tensor_shape: A rank 4 TensorShape.\n\n  Returns:\n    An integer representing the batch size of the tensor.\n  '
    tensor_shape.assert_has_rank(rank=4)
    return get_dim_as_int(tensor_shape[0])

def get_height(tensor_shape):
    if False:
        while True:
            i = 10
    'Returns height from the tensor shape.\n\n  Args:\n    tensor_shape: A rank 4 TensorShape.\n\n  Returns:\n    An integer representing the height of the tensor.\n  '
    tensor_shape.assert_has_rank(rank=4)
    return get_dim_as_int(tensor_shape[1])

def get_width(tensor_shape):
    if False:
        for i in range(10):
            print('nop')
    'Returns width from the tensor shape.\n\n  Args:\n    tensor_shape: A rank 4 TensorShape.\n\n  Returns:\n    An integer representing the width of the tensor.\n  '
    tensor_shape.assert_has_rank(rank=4)
    return get_dim_as_int(tensor_shape[2])

def get_depth(tensor_shape):
    if False:
        for i in range(10):
            print('nop')
    'Returns depth from the tensor shape.\n\n  Args:\n    tensor_shape: A rank 4 TensorShape.\n\n  Returns:\n    An integer representing the depth of the tensor.\n  '
    tensor_shape.assert_has_rank(rank=4)
    return get_dim_as_int(tensor_shape[3])