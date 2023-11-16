import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back, inputs_to_ivy_arrays, handle_numpy_out

@to_ivy_arrays_and_back
@handle_numpy_out
def compress(condition, a, axis=None, out=None):
    if False:
        i = 10
        return i + 15
    condition_arr = ivy.asarray(condition).astype(bool)
    if condition_arr.ndim != 1:
        raise ivy.utils.exceptions.IvyException('Condition must be a 1D array')
    if axis is None:
        arr = ivy.asarray(a).flatten()
        axis = 0
    else:
        arr = ivy.moveaxis(a, axis, 0)
    if condition_arr.shape[0] > arr.shape[0]:
        raise ivy.utils.exceptions.IvyException('Condition contains entries that are out of bounds')
    arr = arr[:condition_arr.shape[0]]
    return ivy.moveaxis(arr[condition_arr], 0, axis)

def diag(v, k=0):
    if False:
        return 10
    return ivy.diag(v, k=k)

@to_ivy_arrays_and_back
def diagonal(a, offset, axis1, axis2):
    if False:
        print('Hello World!')
    return ivy.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)

@to_ivy_arrays_and_back
def fill_diagonal(a, val, wrap=False):
    if False:
        for i in range(10):
            print('nop')
    if a.ndim < 2:
        raise ValueError('array must be at least 2-d')
    end = None
    if a.ndim == 2:
        step = a.shape[1] + 1
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        if not ivy.all(ivy.diff(a.shape) == 0):
            raise ValueError('All dimensions of input must be of equal length')
        step = 1 + ivy.sum(ivy.cumprod(a.shape[:-1]))
    shape = a.shape
    a = ivy.reshape(a, a.size)
    a[:end:step] = val
    a = ivy.reshape(a, shape)

@to_ivy_arrays_and_back
def indices(dimensions, dtype=int, sparse=False):
    if False:
        i = 10
        return i + 15
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,) * N
    if sparse:
        res = ()
    else:
        res = ivy.empty((N,) + dimensions, dtype=dtype)
    for (i, dim) in enumerate(dimensions):
        idx = ivy.arange(dim, dtype=dtype).reshape(shape[:i] + (dim,) + shape[i + 1:])
        if sparse:
            res = res + (idx,)
        else:
            res[i] = idx
    return res

@inputs_to_ivy_arrays
def put_along_axis(arr, indices, values, axis):
    if False:
        return 10
    ivy.put_along_axis(arr, indices, values, axis)

@to_ivy_arrays_and_back
def take_along_axis(arr, indices, axis):
    if False:
        for i in range(10):
            print('nop')
    return ivy.take_along_axis(arr, indices, axis)

@to_ivy_arrays_and_back
def tril_indices(n, k=0, m=None):
    if False:
        return 10
    return ivy.tril_indices(n, m, k)

@to_ivy_arrays_and_back
def unravel_index(indices, shape, order='C'):
    if False:
        while True:
            i = 10
    ret = [x.astype('int64') for x in ivy.unravel_index(indices, shape)]
    return tuple(ret)