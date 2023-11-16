import unittest
import numpy as np
import op_test
import paddle
from paddle.base import core
from paddle.distributed.models.moe import utils

def count(x, upper_num):
    if False:
        i = 10
        return i + 15
    res = np.zeros((upper_num,)).astype(int)
    for i in x.reshape(-1):
        if i >= 0 and i < len(res):
            res[i] += 1
    return res

def number_count_wrapper(numbers, upper_num):
    if False:
        for i in range(10):
            print('nop')
    return paddle._legacy_C_ops.number_count(numbers, 'upper_range', upper_num)

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestNumberCountOpInt64(op_test.OpTest):

    def setUp(self):
        if False:
            return 10
        upper_num = 16
        self.op_type = 'number_count'
        self.python_api = number_count_wrapper
        x = np.random.randint(-1, upper_num, size=(1000, 2)).astype('int64')
        self.inputs = {'numbers': x}
        self.outputs = {'Out': count(x, upper_num)}
        self.attrs = {'upper_range': upper_num}

    def test_forward(self):
        if False:
            while True:
                i = 10
        self.check_output_with_place(paddle.CUDAPlace(0))

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestNumberCountAPI(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.upper_num = 320
        self.x = np.random.randint(-1, self.upper_num, size=(6000, 200)).astype('int64')
        self.out = count(self.x, self.upper_num)
        self.place = paddle.CUDAPlace(0)

    def test_api_static(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x.shape, dtype='int64')
            out = utils._number_count(x, self.upper_num)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x}, fetch_list=[out])
            assert np.allclose(res, self.out)

    def test_api_dygraph(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        out = utils._number_count(x, self.upper_num)
        np.testing.assert_allclose(out.numpy(), self.out)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()