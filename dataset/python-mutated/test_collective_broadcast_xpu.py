import unittest
from get_test_cover_info import XPUOpTestWrapper, create_test_class
from test_collective_base_xpu import TestDistBase
import paddle
from paddle.base import core
paddle.enable_static()

class XPUTestCBroadcastOP(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'c_broadcast'
        self.use_dynamic_create_class = False

    class TestCBroadcastOp(TestDistBase):

        def _setup_config(self):
            if False:
                i = 10
                return i + 15
            pass

        def test_broadcast(self):
            if False:
                while True:
                    i = 10
            self.check_with_place('collective_broadcast_op_xpu.py', 'broadcast', self.in_type_str)
support_types = ['float32']
for stype in support_types:
    create_test_class(globals(), XPUTestCBroadcastOP, stype, ignore_device_version=[core.XPUVersion.XPU1])
if __name__ == '__main__':
    unittest.main()