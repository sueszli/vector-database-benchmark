import numpy as np
import paddle

class TestNewGroupAPI:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        paddle.distributed.init_parallel_env()
        d1 = np.array([1, 2, 3])
        d2 = np.array([2, 3, 4])
        self.tensor1 = paddle.to_tensor(d1)
        self.tensor2 = paddle.to_tensor(d2)

    def test_all(self):
        if False:
            return 10
        gp = paddle.distributed.new_group([0, 1])
        print('gp info:', gp)
        print('test new group api ok')
        tmp = np.array([0, 0, 0])
        result = paddle.to_tensor(tmp)
        paddle.distributed.scatter(result, [self.tensor2, self.tensor1], src=0, group=gp, sync_op=True)
        if gp.rank == 0:
            np.testing.assert_array_equal(result, self.tensor2)
        elif gp.rank == 1:
            np.testing.assert_array_equal(result, self.tensor1)
        print('test scatter api ok')
        paddle.distributed.broadcast(result, src=1, group=gp, sync_op=True)
        np.testing.assert_array_equal(result, self.tensor1)
        print('test broadcast api ok')
        paddle.distributed.reduce(result, dst=0, group=gp, sync_op=True)
        if gp.rank == 0:
            np.testing.assert_array_equal(result, paddle.add(self.tensor1, self.tensor1))
        elif gp.rank == 1:
            np.testing.assert_array_equal(result, self.tensor1)
        print('test reduce api ok')
        paddle.distributed.all_reduce(result, sync_op=True)
        np.testing.assert_array_equal(result, paddle.add(paddle.add(self.tensor1, self.tensor1), self.tensor1))
        print('test all_reduce api ok')
        paddle.distributed.wait(result, gp, use_calc_stream=True)
        paddle.distributed.wait(result, gp, use_calc_stream=False)
        print('test wait api ok')
        result = []
        paddle.distributed.all_gather(result, self.tensor1, group=gp, sync_op=True)
        np.testing.assert_array_equal(result[0], self.tensor1)
        np.testing.assert_array_equal(result[1], self.tensor1)
        print('test all_gather api ok')
        paddle.distributed.barrier(group=gp)
        print('test barrier api ok')
if __name__ == '__main__':
    gpt = TestNewGroupAPI()
    gpt.test_all()