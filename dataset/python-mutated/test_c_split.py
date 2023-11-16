import unittest
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from test_collective_base_xpu import TestDistBase
import paddle
from paddle.base import core
paddle.enable_static()

class XPUTestCSplitOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            print('Hello World!')
        self.op_name = 'c_split'
        self.use_dynamic_create_class = False

    class TestSplitOp(TestDistBase):

        def _setup_config(self):
            if False:
                print('Hello World!')
            pass

        def test_split(self, col_type='c_split'):
            if False:
                return 10
            self.check_with_place('collective_split_op.py', col_type, self.in_type_str)
support_types = get_xpu_op_support_types('c_split')
for stype in support_types:
    create_test_class(globals(), XPUTestCSplitOp, stype, ignore_device_version=[core.XPUVersion.XPU1])
if __name__ == '__main__':
    unittest.main()