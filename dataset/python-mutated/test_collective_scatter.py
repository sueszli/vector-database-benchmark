import unittest
from legacy_test.test_collective_base import TestDistBase
import paddle
paddle.enable_static()

class TestCScatterOp(TestDistBase):

    def _setup_config(self):
        if False:
            print('Hello World!')
        pass

    def test_scatter(self):
        if False:
            return 10
        self.check_with_place('collective_scatter_op.py', 'scatter')
if __name__ == '__main__':
    unittest.main()