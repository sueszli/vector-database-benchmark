from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

def raise_exception():
    if False:
        return 10
    raise RuntimeError('did not expect to be called')

class SmartCondTest(test_util.TensorFlowTestCase):

    def testTrue(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            with session.Session():
                x = constant_op.constant(2)
                y = constant_op.constant(5)
                z = smart_cond.smart_cond(True, lambda : math_ops.multiply(x, 16), lambda : math_ops.multiply(y, 5))
                self.assertEqual(self.evaluate(z), 32)

    def testFalse(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            with session.Session():
                x = constant_op.constant(4)
                y = constant_op.constant(3)
                z = smart_cond.smart_cond(False, lambda : math_ops.multiply(x, 16), lambda : math_ops.multiply(y, 3))
                self.assertEqual(self.evaluate(z), 9)

    def testUnknown(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            with session.Session():
                x = array_ops.placeholder(dtype=dtypes.int32)
                y = smart_cond.smart_cond(x > 0, lambda : constant_op.constant(1), lambda : constant_op.constant(2))
                self.assertEqual(y.eval(feed_dict={x: 1}), 1)
                self.assertEqual(y.eval(feed_dict={x: -1}), 2)

    def testEval(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            with session.Session():
                x = constant_op.constant(1)
                y = constant_op.constant(2)
                z = smart_cond.smart_cond(x * y > 0, lambda : constant_op.constant(1), raise_exception)
                self.assertEqual(z.eval(feed_dict={x: 1}), 1)

    def testPlaceholderWithDefault(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            with session.Session():
                x = array_ops.placeholder_with_default(1, shape=())
                y = smart_cond.smart_cond(x > 0, lambda : constant_op.constant(1), lambda : constant_op.constant(2))
                self.assertEqual(self.evaluate(y), 1)
                self.assertEqual(y.eval(feed_dict={x: -1}), 2)

    def testMissingArg1(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            with session.Session():
                x = constant_op.constant(1)
                with self.assertRaises(TypeError):
                    smart_cond.smart_cond(True, false_fn=lambda : x)

    def testMissingArg2(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            with session.Session():
                x = constant_op.constant(1)
                with self.assertRaises(TypeError):
                    smart_cond.smart_cond(True, lambda : x)

class SmartCaseTest(test_util.TensorFlowTestCase):

    @test_util.run_deprecated_v1
    def testTrue(self):
        if False:
            print('Hello World!')
        x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
        conditions = [(True, lambda : constant_op.constant(1)), (x == 0, raise_exception)]
        y = smart_cond.smart_case(conditions, default=raise_exception, exclusive=False)
        z = smart_cond.smart_case(conditions, default=raise_exception, exclusive=True)
        with session.Session() as sess:
            self.assertEqual(self.evaluate(y), 1)
            self.assertEqual(self.evaluate(z), 1)

    @test_util.run_deprecated_v1
    def testFalse(self):
        if False:
            print('Hello World!')
        conditions = [(False, raise_exception)]
        y = smart_cond.smart_case(conditions, default=lambda : constant_op.constant(1), exclusive=False)
        z = smart_cond.smart_case(conditions, default=lambda : constant_op.constant(1), exclusive=True)
        with session.Session() as sess:
            self.assertEqual(self.evaluate(y), 1)
            self.assertEqual(self.evaluate(z), 1)

    @test_util.run_deprecated_v1
    def testMix(self):
        if False:
            return 10
        x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
        y = constant_op.constant(10)
        conditions = [(x > 1, lambda : constant_op.constant(1)), (y < 1, raise_exception), (False, raise_exception), (True, lambda : constant_op.constant(3))]
        z = smart_cond.smart_case(conditions, default=raise_exception)
        with session.Session() as sess:
            self.assertEqual(sess.run(z, feed_dict={x: 2}), 1)
            self.assertEqual(sess.run(z, feed_dict={x: 0}), 3)

class SmartConstantValueTest(test_util.TensorFlowTestCase):

    def testCond(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            pred = array_ops.placeholder_with_default(True, shape=())
            x = cond.cond(pred, lambda : constant_op.constant(1), lambda : constant_op.constant(2))
            self.assertIsNone(smart_cond.smart_constant_value(x))
if __name__ == '__main__':
    googletest.main()