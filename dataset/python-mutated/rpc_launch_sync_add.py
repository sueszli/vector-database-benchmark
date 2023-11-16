import numpy as np
import paddle
import paddle.distributed as dist
paddle.device.set_device('cpu')

def add(a, b):
    if False:
        while True:
            i = 10
    a = paddle.to_tensor(a, dtype='float32')
    b = paddle.to_tensor(b, dtype='float32')
    res = paddle.add(a, b).numpy()
    return res

def rpc_add(to, args):
    if False:
        for i in range(10):
            print('nop')
    res = dist.rpc.rpc_sync(to, add, args=args)
    return res

def worker_name(rank):
    if False:
        while True:
            i = 10
    return f'worker{rank}'

def main():
    if False:
        while True:
            i = 10
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dist.rpc.init_rpc(worker_name(rank))
    if rank == 0:
        mmap_data1 = np.memmap('rpc_launch_data1.npy', dtype=np.float32, mode='r', shape=(10 * world_size, 100))
        mmap_data2 = np.memmap('rpc_launch_data2.npy', dtype=np.float32, mode='r', shape=(10 * world_size, 100))
        mmap_out = np.memmap('rpc_launch_result.npy', dtype=np.float32, mode='w+', shape=(10 * world_size, 100))
        for i in range(world_size):
            a = mmap_data1[i * 10:(i + 1) * 10, :]
            b = mmap_data2[i * 10:(i + 1) * 10, :]
            args = (a, b)
            out = rpc_add(worker_name(i), args)
            mmap_out[i * 10:(i + 1) * 10, :] = out[:]
    dist.rpc.shutdown()
if __name__ == '__main__':
    main()