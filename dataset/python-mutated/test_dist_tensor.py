import unittest
import numpy as np
import paddle
import paddle.distributed as dist

class TestDistTensor(unittest.TestCase):

    def test_dist_tensor_creation(self):
        if False:
            print('Hello World!')
        shape = [10, 5]
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=['x', 'y'])
        dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None, None])
        dist_tensor_with_numpy = dist.shard_tensor(np.ones(shape, dtype=np.float32), dist_attr=dist_attr)
        dist_tensor_with_tensor = dist.shard_tensor(paddle.ones(shape), dist_attr=dist_attr)
        tensor = paddle.ones(shape)
        self.assertEqual(dist_tensor_with_numpy.shape, shape)
        self.assertEqual(dist_tensor_with_tensor.shape, shape)
        self.assertEqual(dist_tensor_with_numpy.is_dist(), True)
        self.assertEqual(dist_tensor_with_tensor.is_dist(), True)
        self.assertEqual(tensor.is_dist(), False)
        self.assertEqual(str(dist_tensor_with_numpy), str(dist_tensor_with_tensor))
        self.assertEqual(dist_tensor_with_numpy.dist_attr, dist_attr)
        self.assertEqual(dist_tensor_with_tensor.dist_attr, dist_attr)

class TestDistTensorFromFn(unittest.TestCase):

    def run_dtensor_from_fn(self):
        if False:
            for i in range(10):
                print('nop')
        mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None])
        result = dist.dtensor_from_fn(paddle.ones, dist_attr=dist_attr, shape=[16])
        if paddle.in_dynamic_mode():
            dist_attr.dynamic_dims = []
            self.assertIsInstance(result, paddle.Tensor)
            self.assertEqual(result.shape, [16])
            self.assertEqual(result.dist_attr, dist_attr)
        else:
            dist_attr.dynamic_dims = [0]
            self.assertIsInstance(result, paddle.static.Variable)
            self.assertEqual(result.shape, (16,))
            self.assertEqual(result.dist_attr, dist_attr)
        result_zeros = dist.dtensor_from_fn(paddle.zeros, dist_attr=dist_attr, shape=[16])
        if paddle.in_dynamic_mode():
            dist_attr.dynamic_dims = []
            self.assertIsInstance(result, paddle.Tensor)
            self.assertEqual(result.shape, [16])
            self.assertEqual(result.dist_attr, dist_attr)
        else:
            dist_attr.dynamic_dims = [0]
            self.assertIsInstance(result, paddle.static.Variable)
            self.assertEqual(result.shape, (16,))
            self.assertEqual(result.dist_attr, dist_attr)
        result_random = dist.dtensor_from_fn(paddle.rand, dist_attr=dist_attr, shape=[16])
        if paddle.in_dynamic_mode():
            dist_attr.dynamic_dims = []
            self.assertIsInstance(result, paddle.Tensor)
            self.assertEqual(result.shape, [16])
            self.assertEqual(result.dist_attr, dist_attr)
        else:
            dist_attr.dynamic_dims = [0]
            self.assertIsInstance(result, paddle.static.Variable)
            self.assertEqual(result.shape, (16,))
            self.assertEqual(result.dist_attr, dist_attr)
        with self.assertRaises(AssertionError):
            invalid_dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=['x'])
            dist.dtensor_from_fn(paddle.ones, dist_attr=invalid_dist_attr, shape=[2, 3])

    def test_dynamic_mode(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_dtensor_from_fn()

    def test_static_mode(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        self.run_dtensor_from_fn()
        paddle.disable_static()
if __name__ == '__main__':
    unittest.main()