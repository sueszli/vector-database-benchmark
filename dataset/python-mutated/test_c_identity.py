import unittest
from test_collective_base import TestDistBase
import paddle
paddle.enable_static()

class TestIdentityOp(TestDistBase):

    def _setup_config(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_identity(self, col_type='identity'):
        if False:
            return 10
        self.check_with_place('collective_identity_op.py', col_type)
if __name__ == '__main__':
    unittest.main()