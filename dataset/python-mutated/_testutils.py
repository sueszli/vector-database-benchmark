import numpy as np

class _FakeMatrix:

    def __init__(self, data):
        if False:
            print('Hello World!')
        self._data = data
        self.__array_interface__ = data.__array_interface__

class _FakeMatrix2:

    def __init__(self, data):
        if False:
            while True:
                i = 10
        self._data = data

    def __array__(self):
        if False:
            i = 10
            return i + 15
        return self._data

def _get_array(shape, dtype):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get a test array of given shape and data type.\n    Returned NxN matrices are posdef, and 2xN are banded-posdef.\n\n    '
    if len(shape) == 2 and shape[0] == 2:
        x = np.zeros(shape, dtype=dtype)
        x[0, 1:] = -1
        x[1] = 2
        return x
    elif len(shape) == 2 and shape[0] == shape[1]:
        x = np.zeros(shape, dtype=dtype)
        j = np.arange(shape[0])
        x[j, j] = 2
        x[j[:-1], j[:-1] + 1] = -1
        x[j[:-1] + 1, j[:-1]] = -1
        return x
    else:
        np.random.seed(1234)
        return np.random.randn(*shape).astype(dtype)

def _id(x):
    if False:
        i = 10
        return i + 15
    return x

def assert_no_overwrite(call, shapes, dtypes=None):
    if False:
        i = 10
        return i + 15
    '\n    Test that a call does not overwrite its input arguments\n    '
    if dtypes is None:
        dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    for dtype in dtypes:
        for order in ['C', 'F']:
            for faker in [_id, _FakeMatrix, _FakeMatrix2]:
                orig_inputs = [_get_array(s, dtype) for s in shapes]
                inputs = [faker(x.copy(order)) for x in orig_inputs]
                call(*inputs)
                msg = f'call modified inputs [{dtype!r}, {faker!r}]'
                for (a, b) in zip(inputs, orig_inputs):
                    np.testing.assert_equal(a, b, err_msg=msg)