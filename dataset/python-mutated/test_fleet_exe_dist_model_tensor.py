import unittest
import numpy as np
import paddle
from paddle.base.core import DistModelDataType, DistModelTensor
paddle.enable_static()

class TestDistModelTensor(unittest.TestCase):

    def test_dist_model_tensor(self):
        if False:
            return 10
        tensor_32 = np.random.randint(10, 20, size=[20, 2]).astype('int32')
        dist_tensor32 = DistModelTensor(tensor_32, '32_tensor')
        self.assertEqual(dist_tensor32.dtype, DistModelDataType.INT32)
        self.assertEqual(dist_tensor32.data.tolist('int32'), tensor_32.ravel().tolist())
        self.assertEqual(dist_tensor32.data.length(), 40 * 4)
        self.assertEqual(dist_tensor32.name, '32_tensor')
        dist_tensor32.data.reset(tensor_32)
        self.assertEqual(dist_tensor32.as_ndarray().ravel().tolist(), tensor_32.ravel().tolist())
        tensor_64 = np.random.randint(10, 20, size=[20, 2]).astype('int64')
        dist_tensor64 = DistModelTensor(tensor_64, '64_tensor')
        self.assertEqual(dist_tensor64.dtype, DistModelDataType.INT64)
        self.assertEqual(dist_tensor64.data.tolist('int64'), tensor_64.ravel().tolist())
        self.assertEqual(dist_tensor64.data.length(), 40 * 8)
        self.assertEqual(dist_tensor64.name, '64_tensor')
        dist_tensor64.data.reset(tensor_64)
        self.assertEqual(dist_tensor64.as_ndarray().ravel().tolist(), tensor_64.ravel().tolist())
        tensor_float = np.random.randn(20, 2).astype('float32')
        dist_tensor_float = DistModelTensor(tensor_float, 'float_tensor')
        self.assertEqual(dist_tensor_float.dtype, DistModelDataType.FLOAT32)
        self.assertEqual(dist_tensor_float.data.tolist('float32'), tensor_float.ravel().tolist())
        self.assertEqual(dist_tensor_float.data.length(), 40 * 4)
        self.assertEqual(dist_tensor_float.name, 'float_tensor')
        dist_tensor_float.data.reset(tensor_float)
        self.assertEqual(dist_tensor_float.as_ndarray().ravel().tolist(), tensor_float.ravel().tolist())
        tensor_float_16 = np.random.randn(20, 2).astype('float16')
        dist_tensor_float_16 = DistModelTensor(tensor_float_16, 'float_tensor_16')
        self.assertEqual(dist_tensor_float_16.dtype, DistModelDataType.FLOAT16)
        self.assertEqual(dist_tensor_float_16.data.tolist('float16'), tensor_float_16.ravel().tolist())
        self.assertEqual(dist_tensor_float_16.data.length(), 40 * 2)
        self.assertEqual(dist_tensor_float_16.name, 'float_tensor_16')
        dist_tensor_float_16.data.reset(tensor_float_16)
        self.assertEqual(dist_tensor_float_16.as_ndarray().ravel().tolist(), tensor_float_16.ravel().tolist())
if __name__ == '__main__':
    unittest.main()