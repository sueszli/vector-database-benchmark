import cupy
import operator
import numpy
from cupy._core._dtype import get_dtype
supported_dtypes = [get_dtype(x) for x in ('single', 'double', 'csingle', 'cdouble')]
_upcast_memo: dict = {}

def isdense(x):
    if False:
        i = 10
        return i + 15
    return isinstance(x, cupy.ndarray)

def isscalarlike(x):
    if False:
        i = 10
        return i + 15
    'Is x either a scalar, an array scalar, or a 0-dim array?'
    return cupy.isscalar(x) or (isdense(x) and x.ndim == 0)

def get_index_dtype(arrays=(), maxval=None, check_contents=False):
    if False:
        return 10
    'Based on input (integer) arrays ``a``, determines a suitable index data\n    type that can hold the data in the arrays.\n\n    Args:\n        arrays (tuple of array_like):\n            Input arrays whose types/contents to check\n        maxval (float, optional):\n            Maximum value needed\n        check_contents (bool, optional):\n            Whether to check the values in the arrays and not just their types.\n            Default: False (check only the types)\n\n    Returns:\n        dtype: Suitable index data type (int32 or int64)\n    '
    int32min = cupy.iinfo(cupy.int32).min
    int32max = cupy.iinfo(cupy.int32).max
    dtype = cupy.int32
    if maxval is not None:
        if maxval > int32max:
            dtype = cupy.int64
    if isinstance(arrays, cupy.ndarray):
        arrays = (arrays,)
    for arr in arrays:
        arr = cupy.asarray(arr)
        if not cupy.can_cast(arr.dtype, cupy.int32):
            if check_contents:
                if arr.size == 0:
                    continue
                elif cupy.issubdtype(arr.dtype, cupy.integer):
                    maxval = arr.max()
                    minval = arr.min()
                    if minval >= int32min and maxval <= int32max:
                        continue
            dtype = cupy.int64
            break
    return dtype

def validateaxis(axis):
    if False:
        print('Hello World!')
    if axis is not None:
        axis_type = type(axis)
        if axis_type == tuple:
            raise TypeError("Tuples are not accepted for the 'axis' parameter. Please pass in one of the following: {-2, -1, 0, 1, None}.")
        if not cupy.issubdtype(cupy.dtype(axis_type), cupy.integer):
            raise TypeError('axis must be an integer, not {name}'.format(name=axis_type.__name__))
        if not -2 <= axis <= 1:
            raise ValueError('axis out of range')

def upcast(*args):
    if False:
        for i in range(10):
            print('nop')
    "Returns the nearest supported sparse dtype for the\n    combination of one or more types.\n\n    upcast(t0, t1, ..., tn) -> T  where T is a supported dtype\n\n    Examples:\n        >>> upcast('int32')\n        <type 'numpy.int32'>\n        >>> upcast('int32','float32')\n        <type 'numpy.float64'>\n        >>> upcast('bool',float)\n        <type 'numpy.complex128'>\n    "
    t = _upcast_memo.get(args)
    if t is not None:
        return t
    upcast = numpy.result_type(*args)
    for t in supported_dtypes:
        if cupy.can_cast(upcast, t):
            _upcast_memo[args] = t
            return t
    raise TypeError('no supported conversion for types: %r' % (args,))

def check_shape(args, current_shape=None):
    if False:
        print('Hello World!')
    'Check validity of the shape'
    if len(args) == 0:
        raise TypeError("function missing 1 required positional argument: 'shape'")
    elif len(args) == 1:
        try:
            shape_iter = iter(args[0])
        except TypeError:
            new_shape = (operator.index(args[0]),)
        else:
            new_shape = tuple((operator.index(arg) for arg in shape_iter))
    else:
        new_shape = tuple((operator.index(arg) for arg in args))
    if current_shape is None:
        if len(new_shape) != 2:
            raise ValueError('shape must be a 2-tuple of positive integers')
        elif new_shape[0] < 0 or new_shape[1] < 0:
            raise ValueError("'shape' elements cannot be negative")
    else:
        current_size = numpy.prod(current_shape)
        negative_indexes = [i for (i, x) in enumerate(new_shape) if x < 0]
        if len(negative_indexes) == 0:
            new_size = numpy.prod(new_shape)
            if new_size != current_size:
                raise ValueError('cannot reshape array of size {} into shape{}'.format(current_size, new_shape))
        elif len(negative_indexes) == 1:
            skip = negative_indexes[0]
            specified = numpy.prod(new_shape[0:skip] + new_shape[skip + 1:])
            (unspecified, remainder) = divmod(current_size, specified)
            if remainder != 0:
                err_shape = tuple(('newshape' if x < 0 else x for x in new_shape))
                raise ValueError('cannot reshape array of size {} into shape{}'.format(current_size, err_shape))
            new_shape = new_shape[0:skip] + (unspecified,) + new_shape[skip + 1:]
        else:
            raise ValueError('can only specify one unknown dimension')
    if len(new_shape) != 2:
        raise ValueError('matrix shape must be two-dimensional')
    return new_shape