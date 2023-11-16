"""Tests for case statements in XLA."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import test

class CaseTest(xla_test.XLATestCase):

    def testCaseBasic(self):
        if False:
            print('Hello World!')

        @def_function.function(jit_compile=True)
        def switch_case_test(branch_index):
            if False:
                return 10

            def f1():
                if False:
                    for i in range(10):
                        print('nop')
                return array_ops.constant(17)

            def f2():
                if False:
                    return 10
                return array_ops.constant(31)

            def f3():
                if False:
                    for i in range(10):
                        print('nop')
                return array_ops.constant(-1)
            return control_flow_switch_case.switch_case(branch_index, branch_fns={0: f1, 1: f2}, default=f3)
        with ops.device(self.device):
            self.assertEqual(switch_case_test(array_ops.constant(0)).numpy(), 17)
            self.assertEqual(switch_case_test(array_ops.constant(1)).numpy(), 31)
            self.assertEqual(switch_case_test(array_ops.constant(2)).numpy(), -1)
            self.assertEqual(switch_case_test(array_ops.constant(3)).numpy(), -1)

    def testBranchIsPruned(self):
        if False:
            print('Hello World!')

        @def_function.function(jit_compile=True)
        def switch_case_test():
            if False:
                while True:
                    i = 10
            branch_index = array_ops.constant(0)

            def f1():
                if False:
                    return 10
                return array_ops.constant(17)

            def f2():
                if False:
                    return 10
                image_ops.decode_image(io_ops.read_file('/tmp/bmp'))
                return array_ops.constant(31)
            return control_flow_switch_case.switch_case(branch_index, branch_fns={0: f1, 1: f2}, default=f2)
        with ops.device(self.device):
            self.assertEqual(switch_case_test().numpy(), 17)
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()