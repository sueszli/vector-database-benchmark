import unittest
from legacy_test.test_collective_base import TestDistBase
import paddle
paddle.enable_static()

class TestCWaitOp(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        pass

    def test_allreduce_wait(self):
        if False:
            while True:
                i = 10
        self.check_with_place('collective_allreduce_op_wait.py', 'allreduce', check_error_log=True)
if __name__ == '__main__':
    unittest.main()