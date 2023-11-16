import unittest
import numpy as np
import paddle
from paddle import base

class TestFetchVar(unittest.TestCase):

    def set_input(self):
        if False:
            while True:
                i = 10
        self.val = np.array([1, 3, 5]).astype(np.int32)

    def test_fetch_var(self):
        if False:
            i = 10
            return i + 15
        self.set_input()
        x = paddle.tensor.create_tensor(dtype='int32', persistable=True, name='x')
        paddle.assign(self.val, output=x)
        exe = base.Executor(base.CPUPlace())
        exe.run(base.default_main_program(), feed={}, fetch_list=[])
        fetched_x = base.executor._fetch_var('x')
        np.testing.assert_array_equal(fetched_x, self.val)
        self.assertEqual(fetched_x.dtype, self.val.dtype)

class TestFetchNullVar(TestFetchVar):

    def set_input(self):
        if False:
            while True:
                i = 10
        self.val = np.array([]).astype(np.int32)
if __name__ == '__main__':
    unittest.main()