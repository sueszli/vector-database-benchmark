import unittest
import numpy as np
import paddle
from paddle.base import core
from paddle.distributed.models.moe import utils

def limit_by_capacity(expert_count, _capacity, n_worker):
    if False:
        for i in range(10):
            print('nop')
    capacity = np.copy(_capacity)
    old_shape = expert_count.shape
    expert_count = np.reshape(expert_count, (n_worker, len(capacity)))
    output = np.zeros_like(expert_count)
    for wid in range(len(expert_count)):
        for eid in range(len(expert_count[wid])):
            last_cap = capacity[eid]
            if last_cap >= 0:
                capacity[eid] -= expert_count[wid][eid]
            if last_cap >= expert_count[wid][eid]:
                output[wid][eid] = expert_count[wid][eid]
            elif last_cap >= 0:
                output[wid][eid] = last_cap
    return output.reshape(old_shape)

def all_close(exp, out, n_worker):
    if False:
        print('Hello World!')
    exp = exp.reshape(n_worker, -1)
    out = out.reshape(n_worker, -1)
    return np.allclose(exp.sum(0), out.sum(0))

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestLimitByCapacityInt64API(unittest.TestCase):

    def init_test_case(self):
        if False:
            while True:
                i = 10
        self.expert_count = np.random.randint(0, 1000, size=len(self.capacity) * self.n_worker)
        self.out = limit_by_capacity(self.expert_count, self.capacity, self.n_worker)
        self.expert_count = self.expert_count.astype('int64')
        self.capacity = self.capacity.astype('int64')
        self.place = paddle.CUDAPlace(0)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.capacity = np.array([100, 12000, 1200, 800, 4700, 10000, 57, 99])
        self.n_worker = 1024 * 8
        self.init_test_case()

    def test_static_api(self):
        if False:
            return 10
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            capacity = paddle.static.data('capacity', shape=self.capacity.shape, dtype='int64')
            expert_count_tensor = paddle.static.data('ExpertCount', shape=self.expert_count.shape, dtype='int64')
            out = utils._limit_by_capacity(expert_count_tensor, capacity, self.n_worker)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'capacity': self.capacity, 'ExpertCount': self.expert_count}, fetch_list=out)
        assert all_close(self.out, res[0], self.n_worker)

    def test_dygraph_api(self):
        if False:
            print('Hello World!')
        paddle.disable_static(self.place)
        capacity = paddle.to_tensor(self.capacity)
        expert_count_tensor = paddle.to_tensor(self.expert_count)
        out = utils._limit_by_capacity(expert_count_tensor, capacity, self.n_worker)
        assert all_close(self.out, out.numpy(), self.n_worker)

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestLimitByCapacityInt64API_SmallWorker(TestLimitByCapacityInt64API):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.capacity = np.array([100, 12000, 1200, 0, 4700, 1000, 57, 200])
        self.n_worker = 1
        self.init_test_case()
if __name__ == '__main__':
    unittest.main()