import numpy
import chainerx
try:
    import cupy
except Exception:
    cupy = None

class _DummyContext:

    def __enter__(self):
        if False:
            return 10
        pass

    def __exit__(self, type, value, traceback):
        if False:
            return 10
        pass
_dummy_context = _DummyContext()

def _to_numpy(array):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(array, chainerx.ndarray)
    return chainerx.to_numpy(array, copy=False)

def _from_numpy(array):
    if False:
        print('Hello World!')
    assert isinstance(array, numpy.ndarray)
    return chainerx.array(array, copy=False)

def _to_cupy(array):
    if False:
        print('Hello World!')
    assert cupy is not None
    return chainerx._to_cupy(array)

def _from_cupy(array):
    if False:
        while True:
            i = 10
    assert cupy is not None
    assert isinstance(array, cupy.ndarray)
    device = chainerx.get_device('cuda', array.device.id)
    return chainerx._core._fromrawpointer(array.data.mem.ptr, array.shape, array.dtype, array.strides, device, array.data.ptr - array.data.mem.ptr, array)

def _from_chx(array, check_backprop=True):
    if False:
        return 10
    if not isinstance(array, chainerx.ndarray):
        if isinstance(array, numpy.ndarray) or (cupy and isinstance(array, cupy.ndarray)):
            raise TypeError('ChainerX function fallback using NumPy/CuPy arrays is not supported.')
        return (None, _dummy_context, array)
    if check_backprop and array.is_backprop_required():
        raise RuntimeError('ChainerX function fallback using NumPy/CuPy is not supported for arrays that are connected to a graph.')
    backend_name = array.device.backend.name
    if backend_name == 'native':
        return (numpy, _dummy_context, _to_numpy(array))
    if backend_name == 'cuda':
        if cupy is None:
            raise RuntimeError('ChainerX fallback implementation for cuda backend requires cupy to be installed.')
        array_cupy = _to_cupy(array)
        return (cupy, array_cupy.device, array_cupy)
    raise RuntimeError('ChainerX fallback implementation only supports native or cuda backends.')

def _to_chx(array):
    if False:
        print('Hello World!')
    if isinstance(array, numpy.ndarray):
        return _from_numpy(array)
    elif cupy is not None and isinstance(array, cupy.ndarray):
        return _from_cupy(array)
    return array

def _populate_module_functions():
    if False:
        return 10

    def _fix(arr):
        if False:
            return 10
        (xp, dev, arr) = _from_chx(arr)
        with dev:
            ret = xp.fix(arr)
            ret = xp.asarray(ret)
        return _to_chx(ret)

    def _broadcast_arrays(*args):
        if False:
            for i in range(10):
                print('nop')
        (xps, devs, arrs) = zip(*(_from_chx(arr) for arr in args))
        backend = xps[0]
        if not all([xp is backend for xp in xps]):
            raise TypeError('ChainerX function fallback using mixed NumPy/CuPy arrays is not supported.')
        bcasted = backend.broadcast_arrays(*arrs)
        return [_to_chx(ret) for ret in bcasted]

    def _copysign(*args):
        if False:
            return 10
        (xps, devs, arrs) = zip(*(_from_chx(arr) for arr in args))
        backend = xps[0]
        if not all([xp is backend for xp in xps]):
            raise TypeError('ChainerX function fallback using mixed NumPy/CuPy arrays is not supported.')
        with devs[0]:
            y = backend.copysign(*arrs)
        return _to_chx(y)
    chainerx.fix = _fix
    chainerx.broadcast_arrays = _broadcast_arrays
    chainerx.copysign = _copysign

def _populate_ndarray():
    if False:
        print('Hello World!')
    ndarray = chainerx.ndarray
    old_getitem = ndarray.__getitem__

    def __getitem__(arr, key):
        if False:
            i = 10
            return i + 15
        if not isinstance(key, chainerx.ndarray):
            return old_getitem(arr, key)
        is_backprop_required = arr.is_backprop_required()
        (xp, dev, arr) = _from_chx(arr, check_backprop=False)
        if isinstance(key, tuple):
            key = tuple([_from_chx(k, check_backprop=False)[2] for k in key])
        else:
            (_, _, key) = _from_chx(key, check_backprop=False)
        with dev:
            ret = arr[key]
        if is_backprop_required:
            raise RuntimeError('ChainerX getitem fallback for advanced indexing is not supported for arrays that are connected to a graph.')
        return _to_chx(ret)

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        if self.is_backprop_required():
            raise RuntimeError('ChainerX setitem fallback for advanced indexing is not supported for arrays that are connected to a graph.')
        (xp, dev, self) = _from_chx(self)
        if isinstance(key, tuple):
            key = tuple([_from_chx(k)[2] for k in key])
        else:
            (_, _, key) = _from_chx(key)
        (_, _, value) = _from_chx(value)
        with dev:
            self[key] = value
    ndarray.__setitem__ = __setitem__
    ndarray.__getitem__ = __getitem__

    def tolist(arr):
        if False:
            return 10
        (_, dev, arr) = _from_chx(arr)
        with dev:
            ret = arr.tolist()
        return ret
    ndarray.tolist = tolist

def populate():
    if False:
        print('Hello World!')
    _populate_module_functions()
    _populate_ndarray()