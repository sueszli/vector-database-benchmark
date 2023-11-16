import unittest
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from test_collective_base_xpu import TestDistBase
import paddle
from paddle.base import core
paddle.enable_static()

class XPUTestCAllreduceOP(XPUOpTestWrapper):

    def __init__(self):
        if False:
            return 10
        self.op_name = 'c_allreduce_sum'
        self.use_dynamic_create_class = False

    class TestCAllreduceOp(TestDistBase):

        def _setup_config(self):
            if False:
                print('Hello World!')
            pass

        def test_allreduce(self):
            if False:
                print('Hello World!')
            self.check_with_place('collective_allreduce_op_xpu.py', 'allreduce', self.in_type_str)
support_types = get_xpu_op_support_types('c_allreduce_sum')
for stype in support_types:
    create_test_class(globals(), XPUTestCAllreduceOP, stype, ignore_device_version=[core.XPUVersion.XPU1])
if __name__ == '__main__':
    unittest.main()