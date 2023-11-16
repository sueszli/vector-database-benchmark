import unittest
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from test_collective_base_xpu import TestDistBase
import paddle
from paddle.base import core
paddle.enable_static()

class XPUTestCAllgatherOP(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'c_allgather'
        self.use_dynamic_create_class = False

    class TestCAllgatherOp(TestDistBase):

        def _setup_config(self):
            if False:
                while True:
                    i = 10
            pass

        def test_allgather(self):
            if False:
                return 10
            self.check_with_place('collective_allgather_op_xpu.py', 'allgather', self.in_type_str)
support_types = get_xpu_op_support_types('c_allgather')
for stype in support_types:
    create_test_class(globals(), XPUTestCAllgatherOP, stype, ignore_device_version=[core.XPUVersion.XPU1])
if __name__ == '__main__':
    unittest.main()