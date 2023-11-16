import numpy
import chainer
from chainer import _backend
from chainer.backends import _cpu
from chainer.backends import cuda
from chainer.backends import intel64
import chainerx

class ChainerxDevice(_backend.Device):
    """Device for ChainerX backend"""
    xp = chainerx
    supported_array_types = (chainerx.ndarray,)
    __hash__ = _backend.Device.__hash__

    def __init__(self, device: 'chainerx.Device') -> None:
        if False:
            return 10
        assert isinstance(device, chainerx.Device)
        super(ChainerxDevice, self).__init__()
        self.device = device

    @staticmethod
    def from_array(array):
        if False:
            return 10
        if isinstance(array, chainerx.ndarray) and array.device is not None:
            return ChainerxDevice(array.device)
        return None

    @staticmethod
    def from_fallback_device(device):
        if False:
            return 10
        'Returns a :class:`~chainer.backend.ChainerxDevice` corresponding to the fallback device.\n\n        .. seealso::\n            :data:`~chainer.backend.ChainerxDevice.fallback_device`\n        '
        assert isinstance(device, _backend.Device)
        if isinstance(device, _cpu.CpuDevice):
            return ChainerxDevice(chainerx.get_device('native', 0))
        if isinstance(device, cuda.GpuDevice):
            return ChainerxDevice(chainerx.get_device('cuda', device.device.id))
        raise RuntimeError('Only CPU or GPU devices are allowed. Actual: {}'.format(device))

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        return self.device.name

    @property
    def fallback_device(self):
        if False:
            print('Hello World!')
        'Fallback device.\n\n        A fallback device is either a :class:`~chainer.backend.CpuDevice` or\n        a :class:`~chainer.backend.GpuDevice` which shares the same physical\n        device with the original ChainerX device.\n\n        For example, the fallback device of ``native:0`` ChainerX device is\n        :class:`~chainer.backend.CpuDevice`. The fallback device of ``cuda:1``\n        ChainerX device is :class:`~chainer.backend.GpuDevice` with device ID\n        1.\n        '
        backend_name = self.device.backend.name
        if backend_name == 'native':
            return _cpu.CpuDevice()
        if backend_name == 'cuda':
            return cuda.GpuDevice.from_device_id(self.device.index)
        raise RuntimeError("Only 'native' or 'cuda' devices have corresponding fallback devices. Actual: {}".format(backend_name))

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, ChainerxDevice) and other.device == self.device

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<{} {}>'.format(self.__class__.__name__, self.device.name)

    def create_context(self):
        if False:
            return 10
        return chainerx.using_device(self.device)

    def send_array(self, array):
        if False:
            for i in range(10):
                print('nop')
        device = self.device
        if isinstance(array, chainerx.ndarray):
            if array.device is device:
                return array
            return array.to_device(device)
        return _array_to_chainerx(array, device)

    def use(self):
        if False:
            print('Hello World!')
        chainerx.set_default_device(self.device)

    def is_array_supported(self, array):
        if False:
            i = 10
            return i + 15
        return isinstance(array, chainerx.ndarray) and self.device == array.device

def to_chx(array):
    if False:
        i = 10
        return i + 15
    'Converts an array or arrays to ChainerX.\n\n    Destination ChainerX devices are chosen according to the types of input\n    arrays.\n    '
    return _backend._convert_arrays(array, _array_to_chainerx)

def from_chx(array):
    if False:
        while True:
            i = 10
    'Converts an array or arrays from ChainerX to NumPy or CuPy ones.\n\n    Destination array types are chosen such that no copies occur.\n    '
    return _backend._convert_arrays(array, _array_from_chainerx)

def _get_chainerx_device(device_spec):
    if False:
        i = 10
        return i + 15
    if isinstance(device_spec, chainerx.Device):
        return device_spec
    return chainerx.get_device(device_spec)

def _array_to_chainerx(array, device=None):
    if False:
        print('Hello World!')
    assert device is None or isinstance(device, chainerx.Device)
    if array is None:
        return None
    if array.dtype not in chainerx.all_dtypes:
        raise TypeError('Dtype {} is not supported in ChainerX.'.format(array.dtype.name))
    if isinstance(array, chainerx.ndarray):
        if device is None:
            return array
        if device is array.device:
            return array
        return array.to_device(device)
    if isinstance(array, numpy.ndarray):
        if device is None:
            device = chainerx.get_device('native', 0)
        return chainerx.array(array, device=device, copy=False)
    if isinstance(array, cuda.ndarray):
        if device is None:
            device = chainerx.get_device('cuda', array.device.id)
        elif device.backend.name != 'cuda':
            array = _cpu._to_cpu(array)
            return chainerx.array(array, device=device, copy=False)
        elif device.index != array.device.id:
            array = cuda.to_gpu(array, device=device.index)
        return chainerx._core._fromrawpointer(array.data.mem.ptr, array.shape, array.dtype, array.strides, device, array.data.ptr - array.data.mem.ptr, array)
    if isinstance(array, intel64.mdarray):
        return _array_to_chainerx(numpy.array(array), device)
    if numpy.isscalar(array):
        return chainerx.asarray(array)
    raise TypeError('Array cannot be converted into chainerx.ndarray\nActual type: {0}.'.format(type(array)))

def _array_from_chainerx(array):
    if False:
        while True:
            i = 10
    if array is None:
        return None
    if not isinstance(array, chainerx.ndarray):
        if isinstance(array, chainer.get_array_types()):
            return array
        raise TypeError('Tried to convert to a non-ChainerX array from an invalid type: {}'.format(type(array)))
    backend_name = array.device.backend.name
    if backend_name == 'native':
        return _cpu._to_cpu(array)
    if backend_name == 'cuda':
        return cuda.to_gpu(array, array.device.index)
    raise ValueError('Only ChainerX arrays with native or cuda backends can be converted to non-ChainerX arrays.\nActual: {0}.'.format(backend_name))