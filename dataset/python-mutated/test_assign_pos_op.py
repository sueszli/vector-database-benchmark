import unittest
import numpy as np
import op_test
import paddle
from paddle.base import core
from paddle.distributed.models.moe import utils

def assign_pos(x, _cum_count):
    if False:
        i = 10
        return i + 15
    cum_count = np.copy(_cum_count)
    x = x.reshape(-1)
    res = np.zeros((cum_count[-1],), dtype=np.int64)
    for (i, idx) in enumerate(x):
        p = cum_count[idx]
        cum_count[idx] -= 1
        if p >= 1:
            res[p - 1] = i
    return res

def count(x, upper_num):
    if False:
        return 10
    res = np.zeros((upper_num,)).astype(int)
    for i in x.reshape(-1):
        if i >= 0 and i < len(res):
            res[i] += 1
    return res
np_allclose = np.allclose

def assert_allclose(res, out, cum_count):
    if False:
        i = 10
        return i + 15
    c0 = 0
    for c in cum_count:
        if c == c0:
            continue
        data1 = np.copy(res[c0:c])
        data2 = np.copy(out[c0:c])
        data1.sort()
        data2.sort()
        assert np_allclose(data2, data1)
        c0 = c
    return True

def get_redefined_allclose(cum_count):
    if False:
        return 10

    def redefined_allclose(x, y, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return assert_allclose(x, y, cum_count)
    return redefined_allclose

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestAssignPosOpInt64(op_test.OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        x = np.random.randint(0, 16, size=(100, 2)).astype('int64')
        y = count(x, 16)
        cum_count = np.cumsum(y).astype(x.dtype)
        self.op_type = 'assign_pos'
        self.inputs = {'X': x, 'cum_count': cum_count, 'eff_num_len': np.array([cum_count[-1]])}
        self.outputs = {'Out': assign_pos(x, cum_count)}
        self.cum_count = cum_count

    def test_forward(self):
        if False:
            return 10
        paddle.enable_static()
        np.testing.assert_allclose = get_redefined_allclose(self.cum_count)
        self.check_output_with_place(paddle.CUDAPlace(0), check_dygraph=False)

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestAssignPosAPI(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.x = np.random.randint(0, 16, size=(100, 2)).astype('int64')
        y = count(self.x, 16)
        self.cum_count = np.cumsum(y).astype(self.x.dtype)
        self.out = assign_pos(self.x, self.cum_count)
        self.place = paddle.CUDAPlace(0)

    def test_api_static(self):
        if False:
            return 10
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x.shape, dtype='int64')
            cum_count = paddle.static.data('cum_count', self.cum_count.shape, dtype='int64')
            out = utils._assign_pos(x, cum_count)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x, 'cum_count': self.cum_count}, fetch_list=[out])
            assert_allclose(res[0], self.out, self.cum_count)

    def test_api_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        cum_count = paddle.to_tensor(self.cum_count).astype(x.dtype)
        out = utils._assign_pos(x, cum_count)
        assert_allclose(out.numpy(), self.out, self.cum_count)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()