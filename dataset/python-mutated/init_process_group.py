import unittest
import paddle

class TestProcessGroupFp32(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.config()

    def config(self):
        if False:
            return 10
        pass

    def test_init_process_group(self):
        if False:
            print('Hello World!')
        paddle.distributed.init_parallel_env()
        paddle.distributed.new_group()
        group = paddle.distributed.new_group([-1, -2])
        assert group.process_group is None
        group = paddle.distributed.collective.Group(-1, 2, 0, [-1, -2])
        ret = paddle.distributed.barrier(group)
        assert ret is None
        paddle.enable_static()
        in_tensor = paddle.empty((1, 2))
        in_tensor2 = paddle.empty((1, 2))
        paddle.distributed.broadcast(in_tensor, src=0)
        paddle.distributed.all_gather([in_tensor, in_tensor2], in_tensor)
        print('test ok\n')
if __name__ == '__main__':
    unittest.main()