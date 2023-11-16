import random
import sys
import unittest
import numpy as np
import paddle
import paddle.distributed as dist

def init_process_group(strategy=None):
    if False:
        i = 10
        return i + 15
    nranks = paddle.distributed.ParallelEnv().nranks
    rank = dist.ParallelEnv().local_rank
    is_master = True if rank == 0 else False
    pg_group = dist.init_parallel_env()
    return pg_group.process_group

class TestProcessGroupFp32(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        self.config()

    def config(self):
        if False:
            print('Hello World!')
        self.dtype = 'float32'
        self.shape = (2, 10, 5)

    def test_create_process_group_bkcl(self):
        if False:
            while True:
                i = 10
        device_id = paddle.distributed.ParallelEnv().dev_id
        paddle.set_device('xpu:%d' % device_id)
        pg = init_process_group()
        sys.stdout.write(f'rank {pg.rank()}: size {pg.size()} name {pg.name()}\n')
        sys.stdout.write(f'rank {pg.rank()}: test new group api ok\n')
        x = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_y = paddle.to_tensor(y)
        broadcast_result = paddle.assign(tensor_x)
        if pg.rank() == 0:
            task = dist.broadcast(tensor_x, 0)
            paddle.device.xpu.synchronize()
            np.testing.assert_array_equal(broadcast_result, tensor_x)
        else:
            task = dist.broadcast(tensor_y, 0)
            paddle.device.xpu.synchronize()
            np.testing.assert_array_equal(broadcast_result, tensor_y)
        sys.stdout.write(f'rank {pg.rank()}: test broadcast api ok\n')
        if pg.rank() == 0:
            pg.barrier(device_id)
        else:
            task = pg.barrier(device_id)
            task.wait()
        sys.stdout.write(f'rank {pg.rank()}: test barrier api ok\n')
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
            paddle.device.xpu.synchronize()
        else:
            tensor_out_list = [paddle.empty_like(tensor_x), paddle.empty_like(tensor_x)]
            task = dist.all_gather(tensor_out_list, tensor_y)
            paddle.device.xpu.synchronize()
            tensor_out = paddle.concat(tensor_out_list)
        out_1 = paddle.slice(tensor_out, [0], [0], [out_shape[0] // 2])
        out_2 = paddle.slice(tensor_out, [0], [out_shape[0] // 2], [out_shape[0]])
        np.testing.assert_array_equal(tensor_x, out_1)
        np.testing.assert_array_equal(tensor_y, out_2)
        sys.stdout.write(f'rank {pg.rank()}: test allgather api ok\n')
        if pg.rank() == 0:
            task = pg.all_gather(tensor_x, tensor_out)
            task.wait()
            paddle.device.xpu.synchronize()
        else:
            tensor_out_list = []
            task = dist.all_gather(tensor_out_list, tensor_y)
            paddle.device.xpu.synchronize()
            tensor_out = paddle.concat(tensor_out_list)
        out_1 = paddle.slice(tensor_out, [0], [0], [out_shape[0] // 2])
        out_2 = paddle.slice(tensor_out, [0], [out_shape[0] // 2], [out_shape[0]])
        np.testing.assert_array_equal(tensor_x, out_1)
        np.testing.assert_array_equal(tensor_y, out_2)
        sys.stdout.write(f'rank {pg.rank()}: test allgather api2 ok\n')
        x = np.random.random(self.shape).astype(self.dtype)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        tensor_y = paddle.to_tensor(y)
        old_tensor_y = paddle.to_tensor(y)
        sum_result = tensor_x + tensor_y
        if pg.rank() == 0:
            task = dist.reduce(tensor_x, 0, sync_op=True)
            paddle.device.xpu.synchronize()
        else:
            task = dist.reduce(tensor_y, 0, sync_op=False)
            task.wait()
            paddle.device.xpu.synchronize()
        if pg.rank() == 0:
            np.testing.assert_array_equal(tensor_x, sum_result)
        np.testing.assert_array_equal(tensor_y, old_tensor_y)
        sys.stdout.write(f'rank {pg.rank()}: test reduce sum api ok\n')
        in_shape = list(self.shape)
        in_shape[0] *= 2
        x = np.random.random(in_shape).astype(self.dtype)
        y = np.random.random(in_shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        tensor_y = paddle.to_tensor(y)
        need_result = tensor_x + tensor_y
        need_result0 = paddle.slice(need_result, [0], [0], [self.shape[0]])
        need_result1 = paddle.slice(need_result, [0], [self.shape[0]], [in_shape[0]])
        out = np.random.random(self.shape).astype(self.dtype)
        tensor_out = paddle.to_tensor(out)
        if pg.rank() == 0:
            task = dist.reduce_scatter(tensor_out, tensor_x, sync_op=True)
        else:
            task = dist.reduce_scatter(tensor_out, tensor_y, sync_op=False)
            task.wait()
        paddle.device.xpu.synchronize()
        if pg.rank() == 0:
            np.testing.assert_array_equal(need_result0, tensor_out)
        else:
            np.testing.assert_array_equal(need_result1, tensor_out)
        sys.stdout.write(f'rank {pg.rank()}: test reduce_scatter sum api ok\n')
        x = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_y = paddle.to_tensor(y)
        if pg.rank() == 0:
            task = dist.send(tensor_x, 1, sync_op=False)
            task.wait()
        else:
            task = dist.recv(tensor_y, 0, sync_op=False)
            task.wait()
            np.testing.assert_array_equal(tensor_y, tensor_x)
        x = np.random.random(self.shape).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        y = np.random.random(self.shape).astype(self.dtype)
        tensor_y = paddle.to_tensor(y)
        if pg.rank() == 0:
            task = dist.send(tensor_x, 1, sync_op=True)
        else:
            task = dist.recv(tensor_y, 0, sync_op=True)
            np.testing.assert_array_equal(tensor_y, tensor_x)
        x = np.random.uniform(-1, 1, []).astype(self.dtype)
        tensor_x = paddle.to_tensor(x)
        y = np.array(0.2022).astype(self.dtype)
        tensor_y = paddle.to_tensor(y)
        if pg.rank() == 0:
            task = dist.send(tensor_x, 1, sync_op=True)
        else:
            task = dist.recv(tensor_y, 0, sync_op=True)
            assert np.array_equal(tensor_y, tensor_x) and tensor_y.shape == []
        sys.stdout.write(f'rank {pg.rank()}: test send api ok\n')

class TestProcessGroupFp16(TestProcessGroupFp32):

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
        self.dtype = 'float16'
        self.shape = (4, 20, 20)
if __name__ == '__main__':
    unittest.main()