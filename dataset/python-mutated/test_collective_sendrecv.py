import unittest
from legacy_test.test_collective_base import TestDistBase
import paddle
paddle.enable_static()

class TestSendRecvOp(TestDistBase):

    def _setup_config(self):
        if False:
            while True:
                i = 10
        pass

    def test_sendrecv(self):
        if False:
            print('Hello World!')
        self.check_with_place('collective_sendrecv_op.py', 'sendrecv')

    def test_sendrecv_dynamic_shape(self):
        if False:
            return 10
        self.check_with_place('collective_sendrecv_op_dynamic_shape.py', 'sendrecv_dynamic_shape')

    def test_sendrecv_array(self):
        if False:
            return 10
        self.check_with_place('collective_sendrecv_op_array.py', 'sendrecv_array')
if __name__ == '__main__':
    unittest.main()