import unittest
import numpy as np
import paddle
import paddle.distributed as dist

class TestCollectiveAllToAllSingle(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        assert not paddle.distributed.is_initialized(), 'The distributed environment has not been initialized.'
        dist.init_parallel_env()
        assert paddle.distributed.is_initialized(), 'The distributed environment has been initialized.'

    def test_collective_alltoall_single(self):
        if False:
            while True:
                i = 10
        rank = dist.get_rank()
        size = dist.get_world_size()
        input = paddle.ones([size, size], dtype='int64') * rank
        output = paddle.empty([size, size], dtype='int64')
        expected_output = paddle.concat([paddle.ones([1, size], dtype='int64') * i for i in range(size)])
        group = dist.new_group([0, 1])
        dist.alltoall_single(input, output, group=group)
        np.testing.assert_allclose(output.numpy(), expected_output.numpy())
        dist.destroy_process_group(group)
        in_split_sizes = [i + 1 for i in range(size)]
        out_split_sizes = [rank + 1 for i in range(size)]
        input = paddle.ones([sum(in_split_sizes), size], dtype='float32') * rank
        output = paddle.empty([(rank + 1) * size, size], dtype='float32')
        expected_output = paddle.concat([paddle.ones([rank + 1, size], dtype='float32') * i for i in range(size)])
        group = dist.new_group([0, 1])
        task = dist.alltoall_single(input, output, in_split_sizes, out_split_sizes, sync_op=False, group=group)
        task.wait()
        np.testing.assert_allclose(output.numpy(), expected_output.numpy())
        dist.destroy_process_group(group)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        dist.destroy_process_group()
        assert not paddle.distributed.is_initialized(), 'The distributed environment has been deinitialized.'
if __name__ == '__main__':
    unittest.main()