"""Tests for tensorflow.ops.stack_ops."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.platform import test

class StackOpTest(xla_test.XLATestCase):

    def testStackPushPop(self):
        if False:
            return 10
        with self.session(), self.test_scope():
            v = array_ops.placeholder(dtypes.float32)

            def fn():
                if False:
                    while True:
                        i = 10
                h = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name='foo')
                c = gen_data_flow_ops.stack_push_v2(h, v)
                with ops.control_dependencies([c]):
                    c1 = gen_data_flow_ops.stack_pop_v2(h, dtypes.float32)
                return c1
            self.assertAllClose([[4.0, 5.0]], xla.compile(fn)[0].eval({v: [[4.0, 5.0]]}))

    def testStackPushPopSwap(self):
        if False:
            return 10
        with self.session(), self.test_scope():
            a = np.arange(2000)
            x = array_ops.placeholder(dtypes.float32)

            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                h = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name='foo')
                c = gen_data_flow_ops.stack_push_v2(h, x, swap_memory=True)
                with ops.control_dependencies([c]):
                    return gen_data_flow_ops.stack_pop_v2(h, dtypes.float32)
            self.assertAllClose(a, xla.compile(fn)[0].eval({x: a}))

    def testMultiStack(self):
        if False:
            while True:
                i = 10
        with self.session(), self.test_scope():
            v = array_ops.placeholder(dtypes.float32)

            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                h1 = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name='foo')
                c1 = gen_data_flow_ops.stack_push_v2(h1, v)
                with ops.control_dependencies([c1]):
                    c1 = gen_data_flow_ops.stack_pop_v2(h1, dtypes.float32)
                h2 = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name='bar')
                c2 = gen_data_flow_ops.stack_push_v2(h2, 5.0)
                with ops.control_dependencies([c2]):
                    c2 = gen_data_flow_ops.stack_pop_v2(h2, dtypes.float32)
                return c1 + c2
            self.assertAllClose(9.0, xla.compile(fn)[0].eval({v: 4.0}))

    def testSameNameStacks(self):
        if False:
            return 10
        'Different stacks with the same name do not interfere.'
        with self.session() as sess, self.test_scope():
            v1 = array_ops.placeholder(dtypes.float32)
            v2 = array_ops.placeholder(dtypes.float32)

            def fn():
                if False:
                    i = 10
                    return i + 15
                h1 = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name='foo')
                h2 = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name='foo')
                c1 = gen_data_flow_ops.stack_push_v2(h1, v1)
                with ops.control_dependencies([c1]):
                    c2 = gen_data_flow_ops.stack_push_v2(h2, v2)
                with ops.control_dependencies([c2]):
                    pop1 = gen_data_flow_ops.stack_pop_v2(h1, dtypes.float32)
                    pop2 = gen_data_flow_ops.stack_pop_v2(h2, dtypes.float32)
                return [pop1, pop2]
            [pop1_compiled, pop2_compiled] = xla.compile(fn)
            (out1, out2) = sess.run([pop1_compiled, pop2_compiled], {v1: 4.0, v2: 5.0})
            self.assertAllClose(out1, 4.0)
            self.assertAllClose(out2, 5.0)

    def testCloseStack(self):
        if False:
            print('Hello World!')
        with self.session() as sess, self.test_scope():

            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                h = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name='foo')
                gen_data_flow_ops.stack_close_v2(h)
            sess.run(xla.compile(fn))

    def testPushCloseStack(self):
        if False:
            return 10
        with self.session() as sess, self.test_scope():
            v = array_ops.placeholder(dtypes.float32)

            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                h = gen_data_flow_ops.stack_v2(5, dtypes.float32, stack_name='foo')
                c = gen_data_flow_ops.stack_push_v2(h, v)
                with ops.control_dependencies([c]):
                    gen_data_flow_ops.stack_close_v2(h)
            sess.run(xla.compile(fn), {v: [[4.0, 5.0]]})
if __name__ == '__main__':
    test.main()