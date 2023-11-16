import random
import unittest
import numpy as np
import paddle
from paddle.base import core

def init_process_group(strategy=None):
    if False:
        while True:
            i = 10
    nranks = paddle.distributed.ParallelEnv().nranks
    rank = paddle.distributed.ParallelEnv().local_rank
    is_master = True if rank == 0 else False
    store = paddle.base.core.TCPStore('127.0.0.1', 6173, is_master, nranks)
    pg_group = core.ProcessGroupCustom.create(store, paddle.distributed.ParallelEnv().device_type, rank, nranks)
    return pg_group

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
            while True:
                i = 10
        self.dtype = 'float32'
        self.shape = (2, 10, 5)

    def test_create_process_group_xccl(self):
        if False:
            i = 10
            return i + 15
        device_id = paddle.distributed.ParallelEnv().dev_id
        paddle.set_device('custom_cpu:%d' % device_id)
        pg = init_process_group()
        x = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_y = paddle.to_tensor(y)
        sum_result = tensor_x + tensor_y
        if pg.rank() == 0:
            task = pg.all_reduce(tensor_x, core.ReduceOp.SUM, sync_op=True)
            task.wait()
        else:
            task = pg.all_reduce(tensor_y, core.ReduceOp.SUM, sync_op=True)
            task.wait()
        print('test allreduce sum api ok')
        x = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_y = paddle.to_tensor(y)
        max_result = paddle.maximum(tensor_x, tensor_y)
        if pg.rank() == 0:
            task = pg.all_reduce(tensor_x, core.ReduceOp.MAX, sync_op=True)
            task.wait()
        else:
            task = pg.all_reduce(tensor_y, core.ReduceOp.MAX, sync_op=True)
            task.wait()
        print('test allreduce max api ok')
        x = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_y = paddle.to_tensor(y)
        broadcast_result = paddle.assign(tensor_x)
        if pg.rank() == 0:
            task = pg.broadcast(tensor_x, 0, sync_op=True)
            task.wait()
            assert task.is_completed()
        else:
            task = pg.broadcast(tensor_y, 0, sync_op=True)
            task.wait()
            assert task.is_completed()
        print('test broadcast api ok')
        if pg.rank() == 0:
            task = pg.barrier(device_id)
            task.wait()
        else:
            task = pg.barrier(device_id)
            task.wait()
        print('test barrier api ok\n')
        return
        x = np.random.random(self.shape).astype(self.dtype)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        tensor_y = paddle.to_tensor(y)
        out_shape = list(self.shape)
        out_shape[0] *= 2
        out = np.random.random(out_shape).astype(self.dtype)
        tensor_out = paddle.to_tensor(out)
        if pg.rank() == 0:
            task = pg.all_gather(tensor_out, tensor_x, sync_op=True)
            task.wait()
        else:
            task = pg.all_gather(tensor_out, tensor_y, sync_op=True)
            task.wait()
        out_1 = paddle.slice(tensor_out, [0], [0], [out_shape[0] // 2])
        out_2 = paddle.slice(tensor_out, [0], [out_shape[0] // 2], [out_shape[0]])
        print('test allgather api ok\n')
        x = np.random.random(self.shape).astype(self.dtype)
        y = np.random.random(self.shape).astype(self.dtype)
        out1 = np.random.random(self.shape).astype(self.dtype)
        out2 = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        tensor_y = paddle.to_tensor(y)
        tensor_out1 = paddle.to_tensor(out1)
        tensor_out2 = paddle.to_tensor(out2)
        raw_tensor_x_2 = paddle.slice(tensor_x, [0], [self.shape[0] // 2], [self.shape[0]])
        raw_tensor_y_1 = paddle.slice(tensor_y, [0], [0], [self.shape[0] // 2])
        if pg.rank() == 0:
            task = pg.alltoall(tensor_x, tensor_out1)
            task.wait()
        else:
            task = pg.alltoall(tensor_y, tensor_out2)
            task.wait()
        out1_2 = paddle.slice(tensor_out1, [0], [self.shape[0] // 2], [self.shape[0]])
        out2_1 = paddle.slice(tensor_out2, [0], [0], [self.shape[0] // 2])
        print('test alltoall api ok\n')
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
        print('test scatter api ok\n')
if __name__ == '__main__':
    unittest.main()