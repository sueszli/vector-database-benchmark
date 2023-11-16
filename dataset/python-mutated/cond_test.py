"""Tests for tf.cond in XLA."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.client import session
from tensorflow.python.compiler.xla import xla
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test

@test_util.with_control_flow_v2
class CondTest(xla_test.XLATestCase):

    def testCondAndTensorArrayInDefun(self):
        if False:
            i = 10
            return i + 15
        with self.session(), self.test_scope():
            xla_context = control_flow_ops.XLAControlFlowContext()
            xla_context.Enter()

            @def_function.function
            def f():
                if False:
                    return 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
                output = cond.cond(constant_op.constant(True), lambda : ta.write(0, 5.0), lambda : ta.write(0, 10.0))
                return output.stack()
            output_t = f()
            self.assertAllEqual([5.0], self.evaluate(output_t))
            xla_context.Exit()

    def testCondAndTensorArrayInDefun_constFolding(self):
        if False:
            for i in range(10):
                print('nop')
        g = ops.Graph()
        with session.Session(graph=g), g.as_default(), self.test_scope():
            xla_context = control_flow_ops.XLAControlFlowContext()
            xla_context.Enter()

            @def_function.function
            def f():
                if False:
                    i = 10
                    return i + 15
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
                output = cond.cond(constant_op.constant(False), lambda : ta.write(0, 5.0), lambda : ta.write(0, 10.0))
                return output.stack()
            output_t = f()
            self.assertAllEqual([10.0], self.evaluate(output_t))
            xla_context.Exit()

    def testCondAndTensorArray_xlaCompile(self):
        if False:
            return 10
        self.skipTest('b/127846988')
        with self.session(), self.test_scope():
            xla_context = control_flow_ops.XLAControlFlowContext()
            xla_context.Enter()

            def f():
                if False:
                    for i in range(10):
                        print('nop')
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
                output = cond.cond(constant_op.constant(True), lambda : ta.write(0, 5.0), lambda : ta.write(0, 10.0))
                return output.stack()
            (output_t,) = xla.compile(f)
            self.assertAllEqual([5.0], self.evaluate(output_t))
            xla_context.Exit()

    def testCondConstPropagation(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session() as sess, self.test_scope():
            xla_context = control_flow_ops.XLAControlFlowContext()
            xla_context.Enter()
            x = array_ops.placeholder(dtypes.float32)
            p = array_ops.placeholder(dtypes.int32)

            def if_true():
                if False:
                    return 10
                return x[p]

            def if_false():
                if False:
                    for i in range(10):
                        print('nop')
                return 5.0
            output = cond.cond(constant_op.constant(True), if_true, if_false)
            self.assertAllEqual(1.0, sess.run(output, feed_dict={x: [0.0, 1.0, 2.0], p: 1}))
            xla_context.Exit()

    def testCondConstPropagation_xlaCompile(self):
        if False:
            for i in range(10):
                print('nop')
        self.skipTest('b/132430685')
        with self.session(), self.test_scope():
            xla_context = control_flow_ops.XLAControlFlowContext()
            xla_context.Enter()
            x = array_ops.placeholder_with_default([0.0, 1.0, 2.0], shape=[3])
            p = constant_op.constant(1)

            def f():
                if False:
                    for i in range(10):
                        print('nop')

                def if_true():
                    if False:
                        print('Hello World!')
                    return x[p]

                def if_false():
                    if False:
                        return 10
                    return 5.0
                return cond.cond(constant_op.constant(True), if_true, if_false)
            output = xla.compile(f)
            self.assertAllEqual(1.0, self.evaluate(output))
            xla_context.Exit()

    def testCondConstPropagation_errorMsg(self):
        if False:
            i = 10
            return i + 15
        self.skipTest('b/132430685')
        with self.session() as sess, self.test_scope():
            xla_context = control_flow_ops.XLAControlFlowContext()
            xla_context.Enter()
            x = array_ops.placeholder(dtypes.float32)
            p = random_ops.random_uniform([], minval=1, maxval=3, dtype=dtypes.int32)

            def if_true():
                if False:
                    return 10
                return x[:p]

            def if_false():
                if False:
                    while True:
                        i = 10
                return array_ops.fill([p], 5.0)
            output = cond.cond(constant_op.constant(True), if_true, if_false)
            with self.assertRaisesRegex(errors.InvalidArgumentError, 'must be a compile-time constant'):
                sess.run(output, feed_dict={x: [0.0, 1.0, 2.0]})
            xla_context.Exit()

    def testCondConstPropagation_errorMsg_xlaCompile(self):
        if False:
            return 10
        with self.session() as sess, self.test_scope():
            xla_context = control_flow_ops.XLAControlFlowContext()
            xla_context.Enter()
            x = array_ops.placeholder(dtypes.float32)
            p = random_ops.random_uniform([], minval=1, maxval=3, dtype=dtypes.int32)
            condition = math_ops.cast(random_ops.random_uniform([], minval=0, maxval=2, dtype=dtypes.int32), dtypes.bool)

            def f():
                if False:
                    while True:
                        i = 10

                def if_true():
                    if False:
                        print('Hello World!')
                    return x[:p]

                def if_false():
                    if False:
                        for i in range(10):
                            print('nop')
                    return array_ops.fill([p], 5.0)
                return cond.cond(condition, if_true, if_false)
            output = xla.compile(f)
            with self.assertRaisesRegex(errors.InvalidArgumentError, 'must be a compile-time constant'):
                sess.run(output, feed_dict={x: [0.0, 1.0, 2.0]})
            xla_context.Exit()

    def testSwitchCaseAndTensorArrayInDefun(self):
        if False:
            while True:
                i = 10
        self.skipTest('b/127846988')
        with self.session(), self.test_scope():
            xla_context = control_flow_ops.XLAControlFlowContext()
            xla_context.Enter()

            @def_function.function
            def f():
                if False:
                    while True:
                        i = 10
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
                output = control_flow_switch_case.switch_case(constant_op.constant(1), {0: lambda : ta.write(0, 5.0), 1: lambda : ta.write(0, 10.0), 2: lambda : ta.write(0, 15.0)})
                return output.stack()
            output_t = f()
            self.assertAllEqual([10.0], self.evaluate(output_t))
            xla_context.Exit()

    def testSwitchCaseAndTensorArray_xlaCompile(self):
        if False:
            print('Hello World!')
        self.skipTest('b/127846988')
        with self.session(), self.test_scope():
            xla_context = control_flow_ops.XLAControlFlowContext()
            xla_context.Enter()

            def f():
                if False:
                    print('Hello World!')
                ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=1)
                output = control_flow_switch_case.switch_case(constant_op.constant(1), {0: lambda : ta.write(0, 5.0), 1: lambda : ta.write(0, 10.0), 2: lambda : ta.write(0, 15.0)})
                return output.stack()
            (output_t,) = xla.compile(f)
            self.assertAllEqual([10.0], self.evaluate(output_t))
            xla_context.Exit()

    def testSwitchCaseConstPropagation(self):
        if False:
            while True:
                i = 10
        self.skipTest('b/127846988')
        with self.session() as sess, self.test_scope():
            xla_context = control_flow_ops.XLAControlFlowContext()
            xla_context.Enter()
            x = array_ops.placeholder(dtypes.float32)
            p = array_ops.placeholder(dtypes.int32)

            def branch0():
                if False:
                    while True:
                        i = 10
                return 5.0

            def branch1():
                if False:
                    while True:
                        i = 10
                return 15.0

            def branch2():
                if False:
                    print('Hello World!')
                return x[p]
            output = control_flow_switch_case.switch_case(constant_op.constant(2), {0: branch0, 1: branch1, 2: branch2})
            self.assertAllEqual(7.0, sess.run(output, feed_dict={x: [0.0, 1.0, 7.0], p: 2}))
            xla_context.Exit()

    def testCondNoInputs(self):
        if False:
            i = 10
            return i + 15
        'Verifies against `Failed precondition: Expected one input shape`.'
        with self.session(), self.test_scope():
            xla_context = control_flow_ops.XLAControlFlowContext()
            xla_context.Enter()
            for pred in (True, False):
                cond_out = cond.cond(array_ops.placeholder_with_default(pred, []), lambda : constant_op.constant(2.0), lambda : constant_op.constant(1.0))
                self.assertEqual(int(pred) + 1.0, self.evaluate(cond_out))
            xla_context.Exit()
if __name__ == '__main__':
    test.main()