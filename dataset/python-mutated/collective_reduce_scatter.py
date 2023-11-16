import unittest
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed.communication.reduce_scatter import _reduce_scatter_base

class TestCollectiveReduceScatter(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        dist.init_parallel_env()

    def test_collective_reduce_scatter_sum(self):
        if False:
            return 10
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            t1 = paddle.to_tensor([0, 1])
            t2 = paddle.to_tensor([2, 3])
        else:
            t1 = paddle.to_tensor([4, 5])
            t2 = paddle.to_tensor([6, 7])
        input_list = [t1, t2]
        output = paddle.empty(shape=[2], dtype=input_list[0].dtype)
        dist.reduce_scatter(output, input_list)
        if rank == 0:
            np.testing.assert_allclose(output.numpy(), [4, 6])
        elif rank == 1:
            np.testing.assert_allclose(output.numpy(), [8, 10])

    def test_collective_reduce_scatter_max(self):
        if False:
            print('Hello World!')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            t1 = paddle.to_tensor([0, 1], dtype='float16')
            t2 = paddle.to_tensor([2, 3], dtype='float16')
        else:
            t1 = paddle.to_tensor([4, 5], dtype='float16')
            t2 = paddle.to_tensor([6, 7], dtype='float16')
        input_list = [t1, t2]
        output = paddle.empty(shape=[2], dtype=input_list[0].dtype)
        dist.reduce_scatter(output, input_list, op=dist.ReduceOp.MAX)
        if rank == 0:
            np.testing.assert_allclose(output.numpy(), [4, 5])
        elif rank == 1:
            np.testing.assert_allclose(output.numpy(), [6, 7])

    def test_collective_reduce_scatter_base(self):
        if False:
            i = 10
            return i + 15
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        input = paddle.arange(4) + rank
        output = paddle.empty(shape=[2], dtype=input.dtype)
        task = _reduce_scatter_base(output, input, sync_op=False)
        task.wait()
        if rank == 0:
            np.testing.assert_allclose(output.numpy(), [1, 3])
        elif rank == 1:
            np.testing.assert_allclose(output.numpy(), [5, 7])
if __name__ == '__main__':
    unittest.main()