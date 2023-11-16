from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

class FunctionTests(xla_test.XLATestCase):

    def testVarInitializedInFunction(self):
        if False:
            return 10
        with self.test_scope():
            v_holder = []

            @polymorphic_function.function
            def add_var(x):
                if False:
                    return 10
                if not v_holder:
                    v = variables.Variable([1.0, 2.0])
                    v_holder.append(v)
                    already_initialized = variables.Variable(3.0)
                    with ops.init_scope():
                        already_initialized.assign(10.0)
                    v_holder.append(already_initialized)
                return v_holder[0] + v_holder[1] + x
            self.assertAllClose([13.0, 14.0], add_var(constant_op.constant(2.0)))
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()