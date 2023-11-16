import unittest
import numpy as np
import paddle
import paddle.distributed as dist

class TestSemiAutoParallelFunctionalInSingleCard(unittest.TestCase):

    def test_tensor_use_gpudnn(self):
        if False:
            print('Hello World!')
        mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        dense_tensor = paddle.randn([10, 20])
        dist_tensor = dist.shard_tensor(dense_tensor, dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]))
        dist_tensor._use_gpudnn(False)

    def test_tensor_data_ptr(self):
        if False:
            print('Hello World!')
        mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        dense_tensor = paddle.randn([10, 20])
        dist_tensor = dist.shard_tensor(dense_tensor, dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]))
        prt = dist_tensor.data_ptr()

    def test_tensor_offset(self):
        if False:
            while True:
                i = 10
        mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        dense_tensor = paddle.randn([10, 20])
        dist_tensor = dist.shard_tensor(dense_tensor, dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]))
        offset = dist_tensor._offset()

    def test_tensor_copy_to(self):
        if False:
            return 10
        mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        dense_tensor = paddle.randn([10, 20])
        dist_tensor = dist.shard_tensor(dense_tensor, dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]))
        dist_tensor._copy_to(paddle.CUDAPlace(0), True)

    def test_tensor__share_buffer_to(self):
        if False:
            while True:
                i = 10
        mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        dense_tensor = paddle.randn([10, 20])
        dist_tensor = dist.shard_tensor(dense_tensor, dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]))
        dense_tensor2 = paddle.randn([10, 10])
        to = dist.shard_tensor(dense_tensor2, dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]))
        dist_tensor._share_buffer_to(to)

    def test_tensor__is_shared_buffer_with(self):
        if False:
            i = 10
            return i + 15
        mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        dense_tensor = paddle.randn([10, 20])
        dist_tensor = dist.shard_tensor(dense_tensor, dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]))
        dense_tensor2 = paddle.randn([10, 10])
        to = dist.shard_tensor(dense_tensor2, dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]))
        dist_tensor._share_buffer_to(to)
        self.assertTrue(dist_tensor._is_shared_buffer_with(to))

    def test_tensor_strides(self):
        if False:
            for i in range(10):
                print('nop')
        mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        dense_tensor = paddle.randn([10, 20])
        dense_tensor = dense_tensor.reshape([20, 10])
        dist_tensor = dist.shard_tensor(dense_tensor, dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]))
        strides = dist_tensor.get_strides()
        is_contiguous = dist_tensor.is_contiguous()
        dist_tensor = dist_tensor.contiguous()

    def test_tensor_uva(self):
        if False:
            while True:
                i = 10
        mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        place = paddle.CPUPlace()
        np_value = np.random.random(size=[10, 30]).astype('float32')
        dense_tensor = paddle.to_tensor(np_value, place=place)
        dist_tensor = dist.shard_tensor(dense_tensor, place=place, dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]))
        dist_tensor._uva()

    def test_tensor_properties(self):
        if False:
            print('Hello World!')
        mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        dense_tensor = paddle.randn([10, 20])
        dense_tensor = dense_tensor.reshape([20, 10])
        dist_tensor = dist.shard_tensor(dense_tensor, dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]))
        type = dist_tensor.type
        strides = dist_tensor.strides
        offsets = dist_tensor.offset

    def test_tensor_set_data(self):
        if False:
            i = 10
            return i + 15
        mesh = dist.ProcessMesh([0, 1], dim_names=['x'])
        dense_tensor_a = paddle.randn([10, 20])
        dist_tensor_a = dist.shard_tensor(dense_tensor_a, dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]))
        dense_tensor_b = paddle.randn([5, 8])
        dist_tensor_b = dist.shard_tensor(dense_tensor_b, dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, None]))
        dist_tensor_b.data = dist_tensor_a
if __name__ == '__main__':
    unittest.main()