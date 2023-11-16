import random
import unittest
from copy import deepcopy
import numpy as np
import paddle
from paddle.base import core

class TestProcessGroupFp32(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        self.config()

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = 'float32'
        self.shape = (2, 10, 5)

    def test_create_process_group_gloo(self):
        if False:
            i = 10
            return i + 15
        nranks = paddle.distributed.ParallelEnv().nranks
        rank = paddle.distributed.ParallelEnv().local_rank
        is_master = True if rank == 0 else False
        store = paddle.base.core.TCPStore('127.0.0.1', 6272, is_master, nranks, 30)
        pg = paddle.base.core.ProcessGroupGloo.create(store, rank, nranks)
        paddle.device.set_device('cpu')
        x = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_y = paddle.to_tensor(y)
        sum_result = x + y
        if rank == 0:
            task = pg.allreduce(tensor_x)
            task.wait()
            np.testing.assert_equal(tensor_x, sum_result)
        else:
            task = pg.allreduce(tensor_y)
            task.wait()
            np.testing.assert_equal(tensor_y, sum_result)
        print('test allreduce sum api ok')
        x = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_y = paddle.to_tensor(y)
        max_result = paddle.maximum(tensor_x, tensor_y)
        if rank == 0:
            task = pg.allreduce(tensor_x, core.ReduceOp.MAX)
            task.wait()
            np.testing.assert_array_equal(tensor_x, max_result)
        else:
            task = pg.allreduce(tensor_y, core.ReduceOp.MAX)
            task.wait()
            np.testing.assert_array_equal(tensor_y, max_result)
        print('test allreduce max api ok')
        x = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_y = paddle.to_tensor(y)
        broadcast_result = paddle.assign(tensor_x)
        if rank == 0:
            task = pg.broadcast(tensor_x, 0)
            np.testing.assert_array_equal(broadcast_result, tensor_x)
        else:
            task = pg.broadcast(tensor_y, 0)
            np.testing.assert_array_equal(broadcast_result, tensor_y)
        print('test broadcast api ok')
        x = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_y_1 = paddle.to_tensor(y)
        tensor_y_2 = deepcopy(tensor_y_1)
        send_recv_result_1 = paddle.assign(tensor_x)
        send_recv_result_2 = paddle.assign(tensor_y_2)
        if pg.rank() == 0:
            task = pg.send(tensor_x, pg.size() - 1, True)
        elif pg.rank() == pg.size() - 1:
            task = pg.recv(tensor_y_1, 0, True)
            np.testing.assert_array_equal(send_recv_result_1, tensor_y_1)
        if pg.rank() == 0:
            task = pg.recv(tensor_x, pg.size() - 1, True)
            np.testing.assert_array_equal(send_recv_result_2, tensor_x)
        elif pg.rank() == pg.size() - 1:
            task = pg.send(tensor_y_2, 0, True)
        print('test send_recv api ok')
        if pg.rank() == 0:
            task = pg.barrier()
            task.wait()
        else:
            task = pg.barrier()
            task.wait()
        print('test barrier api ok\n')
        x = np.random.random(self.shape).astype(self.dtype)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        tensor_y = paddle.to_tensor(y)
        out_shape = list(self.shape)
        out_shape[0] *= 2
        out = np.random.random(out_shape).astype(self.dtype)
        tensor_out = paddle.to_tensor(out)
        if pg.rank() == 0:
            task = pg.all_gather(tensor_x, tensor_out)
            task.wait()
            paddle.device.cuda.synchronize()
        else:
            task = pg.all_gather(tensor_y, tensor_out)
            task.wait()
        out_1 = paddle.slice(tensor_out, [0], [0], [out_shape[0] // 2])
        out_2 = paddle.slice(tensor_out, [0], [out_shape[0] // 2], [out_shape[0]])
        np.testing.assert_array_equal(tensor_x, out_1)
        np.testing.assert_array_equal(tensor_y, out_2)
        print('test allgather api ok\n')
        x = np.random.random(self.shape).astype(self.dtype)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        tensor_y = paddle.to_tensor(y)
        sum_result = tensor_x + tensor_y
        if pg.rank() == 0:
            task = pg.reduce(tensor_x, 0)
            task.wait()
        else:
            task = pg.reduce(tensor_y, 0)
            task.wait()
        if pg.rank() == 0:
            np.testing.assert_array_equal(tensor_x, sum_result)
        print('test reduce sum api ok\n')
        in_shape = list(self.shape)
        in_shape[0] *= 2
        x = np.random.random(in_shape).astype(self.dtype)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        tensor_y = paddle.to_tensor(y)
        if pg.rank() == 0:
            task = pg.scatter(tensor_x, tensor_y, 0)
            task.wait()
        else:
            task = pg.scatter(tensor_x, tensor_y, 0)
            task.wait()
        out1 = paddle.slice(tensor_x, [0], [0], [self.shape[0]])
        out2 = paddle.slice(tensor_x, [0], [self.shape[0]], [self.shape[0] * 2])
        if pg.rank() == 0:
            np.testing.assert_array_equal(tensor_y, out1)
        else:
            np.testing.assert_array_equal(tensor_y, out2)
        print('test scatter api ok\n')

        def test_gather(root):
            if False:
                return 10
            tensor_x = [paddle.zeros(self.shape).astype(self.dtype) for _ in range(pg.size())]
            tensor_y = [paddle.to_tensor(np.random.random(self.shape).astype(self.dtype)) for _ in range(pg.size())]
            if pg.rank() == root:
                task = pg.gather(tensor_y[root], tensor_x, root, True)
                task.wait()
                np.testing.assert_array_equal(tensor_x, tensor_y)
            else:
                task = pg.gather(tensor_y[pg.rank()], tensor_x, root, True)
                task.wait()
        test_gather(0)
        test_gather(pg.size() - 1)
        print('test gather api ok\n')
if __name__ == '__main__':
    unittest.main()