import unittest
from paddle.base import core
from paddle.device.cuda import device_count, get_device_properties

class TestGetDeviceProperties(unittest.TestCase):

    def test_get_device_properties_default(self):
        if False:
            for i in range(10):
                print('nop')
        if core.is_compiled_with_cuda():
            props = get_device_properties()
            self.assertIsNotNone(props)

    def test_get_device_properties_str(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            props = get_device_properties('gpu:0')
            self.assertIsNotNone(props)

    def test_get_device_properties_int(self):
        if False:
            while True:
                i = 10
        if core.is_compiled_with_cuda():
            gpu_num = device_count()
            for i in range(gpu_num):
                props = get_device_properties(i)
                self.assertIsNotNone(props)

    def test_get_device_properties_CUDAPlace(self):
        if False:
            print('Hello World!')
        if core.is_compiled_with_cuda():
            device = core.CUDAPlace(0)
            props = get_device_properties(device)
            self.assertIsNotNone(props)

class TestGetDevicePropertiesError(unittest.TestCase):

    def test_error_api(self):
        if False:
            return 10
        if core.is_compiled_with_cuda():

            def test_device_indexError_error():
                if False:
                    return 10
                device_error = device_count() + 1
                props = get_device_properties(device_error)
            self.assertRaises(IndexError, test_device_indexError_error)

            def test_device_value_error1():
                if False:
                    while True:
                        i = 10
                device_error = 'gpu1'
                props = get_device_properties(device_error)
            self.assertRaises(ValueError, test_device_value_error1)

            def test_device_value_error2():
                if False:
                    i = 10
                    return i + 15
                device_error = float(device_count())
                props = get_device_properties(device_error)
            self.assertRaises(ValueError, test_device_value_error2)
if __name__ == '__main__':
    unittest.main()