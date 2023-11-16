import unittest
import numpy as np
import paddle
import paddle.distributed as dist

class TestCollectiveBatchIsendIrecv(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        dist.init_parallel_env()

    def test_collective_batch_isend_irecv(self):
        if False:
            for i in range(10):
                print('nop')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        send_t = paddle.arange(2) + rank
        recv_t = paddle.empty(shape=[2], dtype=send_t.dtype)
        send_op = dist.P2POp(dist.isend, send_t, (rank + 1) % world_size)
        recv_op = dist.P2POp(dist.irecv, recv_t, (rank - 1 + world_size) % world_size)
        tasks = dist.batch_isend_irecv([send_op, recv_op])
        for task in tasks:
            task.wait()
        if rank == 0:
            np.testing.assert_allclose(recv_t.numpy(), [1, 2])
        elif rank == 1:
            np.testing.assert_allclose(recv_t.numpy(), [0, 1])
if __name__ == '__main__':
    unittest.main()