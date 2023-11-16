import unittest
import paddle
from paddle import base

class TensorToTest(unittest.TestCase):

    def test_Tensor_to_dtype(self):
        if False:
            i = 10
            return i + 15
        tensorx = paddle.to_tensor([1, 2, 3])
        valid_dtypes = ['bfloat16', 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'complex64', 'complex128', 'bool']
        for dtype in valid_dtypes:
            tensorx = tensorx.to(dtype)
            typex_str = str(tensorx.dtype)
            self.assertTrue(typex_str, 'paddle.' + dtype)

    def test_Tensor_to_device(self):
        if False:
            return 10
        tensorx = paddle.to_tensor([1, 2, 3])
        places = ['cpu']
        if base.core.is_compiled_with_cuda():
            places.append('gpu:0')
            places.append('gpu')
        for place in places:
            tensorx = tensorx.to(place)
            placex_str = str(tensorx.place)
            if place == 'gpu':
                self.assertTrue(placex_str, 'Place(' + place + ':0)')
            else:
                self.assertTrue(placex_str, 'Place(' + place + ')')

    def test_Tensor_to_device_dtype(self):
        if False:
            i = 10
            return i + 15
        tensorx = paddle.to_tensor([1, 2, 3])
        places = ['cpu']
        if base.core.is_compiled_with_cuda():
            places.append('gpu:0')
            places.append('gpu')
        valid_dtypes = ['bfloat16', 'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'complex64', 'complex128', 'bool']
        for dtype in valid_dtypes:
            for place in places:
                tensorx = tensorx.to(place, dtype)
                placex_str = str(tensorx.place)
                if place == 'gpu':
                    self.assertTrue(placex_str, 'Place(' + place + ':0)')
                else:
                    self.assertTrue(placex_str, 'Place(' + place + ')')
                typex_str = str(tensorx.dtype)
                self.assertTrue(typex_str, 'paddle.' + dtype)

    def test_Tensor_to_blocking(self):
        if False:
            return 10
        tensorx = paddle.to_tensor([1, 2, 3])
        tensorx = tensorx.to('cpu', 'int32', False)
        placex_str = str(tensorx.place)
        self.assertTrue(placex_str, 'Place(cpu)')
        typex_str = str(tensorx.dtype)
        self.assertTrue(typex_str, 'paddle.int32')
        tensor2 = paddle.to_tensor([4, 5, 6])
        tensor2 = tensor2.to(tensorx, False)
        place2_str = str(tensor2.place)
        self.assertTrue(place2_str, 'Place(cpu)')
        type2_str = str(tensor2.dtype)
        self.assertTrue(type2_str, 'paddle.int32')
        tensor2 = tensor2.to('float16', False)
        type2_str = str(tensor2.dtype)
        self.assertTrue(type2_str, 'paddle.float16')

    def test_Tensor_to_other(self):
        if False:
            print('Hello World!')
        tensor1 = paddle.to_tensor([1, 2, 3], dtype='int8', place='cpu')
        tensor2 = paddle.to_tensor([1, 2, 3])
        tensor2 = tensor2.to(tensor1)
        self.assertTrue(tensor2.dtype, tensor1.dtype)
        self.assertTrue(type(tensor2.place), type(tensor1.place))

    def test_kwargs(self):
        if False:
            print('Hello World!')
        tensorx = paddle.to_tensor([1, 2, 3])
        tensorx = tensorx.to(device='cpu', dtype='int8', blocking=True)
        placex_str = str(tensorx.place)
        self.assertTrue(placex_str, 'Place(cpu)')
        typex_str = str(tensorx.dtype)
        self.assertTrue(typex_str, 'paddle.int8')
        tensor2 = paddle.to_tensor([4, 5, 6])
        tensor2 = tensor2.to(other=tensorx)
        place2_str = str(tensor2.place)
        self.assertTrue(place2_str, 'Place(cpu)')
        type2_str = str(tensor2.dtype)
        self.assertTrue(type2_str, 'paddle.int8')

    def test_error(self):
        if False:
            print('Hello World!')
        tensorx = paddle.to_tensor([1, 2, 3])
        try:
            tensorx = tensorx.to('error_device')
        except Exception as error:
            self.assertIsInstance(error, ValueError)
        try:
            tensorx = tensorx.to('cpu', 'int32', False, 'test_aug')
        except Exception as error:
            self.assertIsInstance(error, TypeError)
        try:
            tensorx = tensorx.to('cpu', 'int32', test_key=False)
        except Exception as error:
            self.assertIsInstance(error, TypeError)
if __name__ == '__main__':
    unittest.main()