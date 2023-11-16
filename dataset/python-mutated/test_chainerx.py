import unittest
import numpy
import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr
import chainerx

@testing.inject_backend_tests(None, [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestChainerxDevice(unittest.TestCase):

    def check_device(self, device, backend_config):
        if False:
            print('Hello World!')
        assert isinstance(device, backend.ChainerxDevice)
        assert device.xp is chainerx
        assert device.supported_array_types == (chainerx.ndarray,)
        assert device.name == backend_config.chainerx_device
        assert str(device) == backend_config.chainerx_device
        assert isinstance(hash(device), int)
        chainerx_device_comps = backend_config.chainerx_device.split(':')
        if chainerx_device_comps[0] == 'native':
            assert isinstance(device.fallback_device, backend.CpuDevice)
        elif chainerx_device_comps[0] == 'cuda':
            assert isinstance(device.fallback_device, backend.GpuDevice)
            assert device.fallback_device.device.id == int(chainerx_device_comps[1])
        else:
            assert False

    def test_init(self, backend_config):
        if False:
            print('Hello World!')
        name = backend_config.chainerx_device
        chx_device = chainerx.get_device(name)
        device = backend.ChainerxDevice(chx_device)
        self.check_device(device, backend_config)
        assert device.device is chx_device

    def test_from_array(self, backend_config):
        if False:
            for i in range(10):
                print('nop')
        arr = backend_config.get_array(numpy.ndarray((2,), numpy.float32))
        assert arr.device.name == backend_config.chainerx_device
        expected_device = backend_config.device
        device = backend.ChainerxDevice.from_array(arr)
        self.check_device(device, backend_config)
        assert device == expected_device
        device = backend.get_device_from_array(arr)
        self.check_device(device, backend_config)
        assert device == expected_device

    def test_from_fallback_device(self, backend_config):
        if False:
            return 10
        tmp_device = backend.ChainerxDevice(chainerx.get_device(backend_config.chainerx_device))
        fallback_device = tmp_device.fallback_device
        device = backend.ChainerxDevice.from_fallback_device(fallback_device)
        self.check_device(device, backend_config)
        assert device.fallback_device == fallback_device

@testing.inject_backend_tests(None, [{}, {'use_cuda': True}])
class TestChainerxDeviceFromArrayInvalidArray(unittest.TestCase):

    def test_from_array(self, backend_config):
        if False:
            while True:
                i = 10
        arr = backend_config.get_array(numpy.ndarray((2,), numpy.float32))
        device = backend.ChainerxDevice.from_array(arr)
        assert device is None

@testing.parameterize(*testing.product({'value': [None, 1, ()]}))
class TestChainerxDeviceFromArrayInvalidValue(unittest.TestCase):

    def test_from_array(self):
        if False:
            while True:
                i = 10
        device = backend.ChainerxDevice.from_array(self.value)
        assert device is None

@testing.inject_backend_tests(None, [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestChainerxDeviceUse(unittest.TestCase):

    def test_use(self, backend_config):
        if False:
            i = 10
            return i + 15
        device = chainer.get_device(backend_config.chainerx_device)
        with chainerx.using_device('native:1'):
            device.use()
            assert device.device is chainerx.get_default_device()

@chainer.testing.inject_backend_tests(None, [{}, {'use_cuda': True, 'cuda_device': 0}, {'use_cuda': True, 'cuda_device': 1}, {'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
@attr.chainerx
class TestFromToChainerx(unittest.TestCase):

    def check_equal_memory_shared(self, arr1, arr2):
        if False:
            for i in range(10):
                print('nop')
        numpy.testing.assert_array_equal(backend.CpuDevice().send(arr1), backend.CpuDevice().send(arr2))
        with chainer.using_device(backend.get_device_from_array(arr1)):
            arr1 += 2
        numpy.testing.assert_array_equal(backend.CpuDevice().send(arr1), backend.CpuDevice().send(arr2))
        with chainer.using_device(backend.get_device_from_array(arr1)):
            arr1 -= 2

    def test_from_chx(self, backend_config):
        if False:
            for i in range(10):
                print('nop')
        arr = backend_config.get_array(numpy.ones((2, 3), numpy.float32))
        arr_converted = backend.from_chx(arr)
        src_device = backend_config.device
        if src_device.xp is chainerx:
            dst_xp = src_device.fallback_device.xp
            assert isinstance(arr_converted, dst_xp.ndarray)
            if dst_xp is cuda.cupy:
                assert arr_converted.device.id == src_device.device.index
        else:
            assert arr is arr_converted
        with backend_config:
            self.check_equal_memory_shared(arr, arr_converted)

    def test_to_chx(self, backend_config):
        if False:
            print('Hello World!')
        arr = backend_config.get_array(numpy.ones((2, 3), numpy.float32))
        arr_converted = backend.to_chx(arr)
        src_device = backend_config.device
        assert isinstance(arr_converted, chainerx.ndarray)
        if src_device.xp is chainerx:
            assert arr is arr_converted
        elif src_device.xp is cuda.cupy:
            assert arr.device.id == arr_converted.device.index
        self.check_equal_memory_shared(arr, arr_converted)

@chainer.testing.inject_backend_tests(None, [{}, {'use_cuda': True, 'cuda_device': 0}, {'use_cuda': True, 'cuda_device': 1}, {'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
@chainer.testing.inject_backend_tests(None, [{'use_chainerx': True, 'chainerx_device': 'native:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:0'}, {'use_chainerx': True, 'chainerx_device': 'cuda:1'}])
class TestChainerxIsArraySupported(unittest.TestCase):

    def test_is_array_supported(self, backend_config1, backend_config2):
        if False:
            print('Hello World!')
        target = backend_config1.device
        arr = backend_config2.get_array(numpy.ndarray((2,), numpy.float32))
        device = backend_config2.device
        if isinstance(device, backend.ChainerxDevice) and device.device == target.device:
            assert target.is_array_supported(arr)
        else:
            assert not target.is_array_supported(arr)
testing.run_module(__name__, __file__)