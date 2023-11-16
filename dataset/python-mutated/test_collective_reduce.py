import unittest
from test_collective_base import TestDistBase
import paddle
paddle.enable_static()

class TestCReduceOp(TestDistBase):

    def _setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_reduce(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_with_place('collective_reduce_op.py', 'reduce')

    def test_reduce_calc_stream(self):
        if False:
            while True:
                i = 10
        self.check_with_place('collective_reduce_op_calc_stream.py', 'reduce')
if __name__ == '__main__':
    unittest.main()