"""A library of common shape functions."""
import itertools
from tensorflow.python.framework import tensor_shape

def _broadcast_shape_helper(shape_x, shape_y):
    if False:
        print('Hello World!')
    'Helper functions for is_broadcast_compatible and broadcast_shape.\n\n  Args:\n    shape_x: A `TensorShape`\n    shape_y: A `TensorShape`\n\n  Returns:\n    Returns None if the shapes are not broadcast compatible,\n    a list of the broadcast dimensions otherwise.\n  '
    broadcasted_dims = reversed(list(itertools.zip_longest(reversed(shape_x.dims), reversed(shape_y.dims), fillvalue=tensor_shape.Dimension(1))))
    return_dims = []
    for (dim_x, dim_y) in broadcasted_dims:
        if dim_x.value is None or dim_y.value is None:
            if dim_x.value is not None and dim_x.value > 1:
                return_dims.append(dim_x)
            elif dim_y.value is not None and dim_y.value > 1:
                return_dims.append(dim_y)
            else:
                return_dims.append(None)
        elif dim_x.value == 1:
            return_dims.append(dim_y)
        elif dim_y.value == 1:
            return_dims.append(dim_x)
        elif dim_x.value == dim_y.value:
            return_dims.append(dim_x.merge_with(dim_y))
        else:
            return None
    return return_dims

def is_broadcast_compatible(shape_x, shape_y):
    if False:
        return 10
    'Returns True if `shape_x` and `shape_y` are broadcast compatible.\n\n  Args:\n    shape_x: A `TensorShape`\n    shape_y: A `TensorShape`\n\n  Returns:\n    True if a shape exists that both `shape_x` and `shape_y` can be broadcasted\n    to.  False otherwise.\n  '
    if shape_x.ndims is None or shape_y.ndims is None:
        return False
    return _broadcast_shape_helper(shape_x, shape_y) is not None

def broadcast_shape(shape_x, shape_y):
    if False:
        i = 10
        return i + 15
    'Returns the broadcasted shape between `shape_x` and `shape_y`.\n\n  Args:\n    shape_x: A `TensorShape`\n    shape_y: A `TensorShape`\n\n  Returns:\n    A `TensorShape` representing the broadcasted shape.\n\n  Raises:\n    ValueError: If the two shapes can not be broadcasted.\n  '
    if shape_x.ndims is None or shape_y.ndims is None:
        return tensor_shape.unknown_shape()
    return_dims = _broadcast_shape_helper(shape_x, shape_y)
    if return_dims is None:
        raise ValueError(f'Incompatible shapes for broadcasting. Two shapes are compatible if for each dimension pair they are either equal or one of them is 1. Received: {shape_x} and {shape_y}.')
    return tensor_shape.TensorShape(return_dims)