import unittest
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from test_collective_base_xpu import TestDistBase
import paddle
from paddle.base import core
paddle.enable_static()

class XPUTestCConcatOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'c_concat'
        self.use_dynamic_create_class = False

    class TestConcatOp(TestDistBase):

        def _setup_config(self):
            if False:
                while True:
                    i = 10
            pass

        def test_concat(self, col_type='c_concat'):
            if False:
                for i in range(10):
                    print('nop')
            self.check_with_place('collective_concat_op.py', col_type, self.in_type_str)
support_types = get_xpu_op_support_types('c_concat')
for stype in support_types:
    create_test_class(globals(), XPUTestCConcatOp, stype, ignore_device_version=[core.XPUVersion.XPU1])
if __name__ == '__main__':
    unittest.main()