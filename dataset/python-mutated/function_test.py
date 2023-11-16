"""Test cases for Tensorflow functions."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest

class FunctionTest(xla_test.XLATestCase):

    def testFunction(self):
        if False:
            return 10
        'Executes a simple TensorFlow function.'

        def APlus2B(a, b):
            if False:
                while True:
                    i = 10
            return a + b * 2
        aval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
        bval = np.array([5, 6, 7, 8]).reshape([2, 2]).astype(np.float32)
        expected = APlus2B(aval, bval)
        with self.session():

            @function.Defun(dtypes.float32, dtypes.float32)
            def Foo(a, b):
                if False:
                    i = 10
                    return i + 15
                return APlus2B(a, b)
            a = constant_op.constant(aval, name='a')
            b = constant_op.constant(bval, name='b')
            with self.test_scope():
                call_f = Foo(a, b)
            result = self.evaluate(call_f)
        self.assertAllClose(result, expected, rtol=0.001)

    def testNestedFunctions(self):
        if False:
            print('Hello World!')
        'Executes two nested TensorFlow functions.'

        def TimesTwo(x):
            if False:
                while True:
                    i = 10
            return x * 2

        def APlus2B(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return a + TimesTwo(b)
        aval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
        bval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
        expected = APlus2B(aval, bval)
        with self.session():

            @function.Defun(dtypes.float32, dtypes.float32)
            def Foo(a, b):
                if False:
                    i = 10
                    return i + 15
                return APlus2B(a, b)
            a = constant_op.constant(aval, name='a')
            b = constant_op.constant(bval, name='b')
            with self.test_scope():
                call_g = Foo(a, b)
            result = self.evaluate(call_g)
        self.assertAllClose(result, expected, rtol=0.001)

    def testFunctionMultipleRetvals(self):
        if False:
            print('Hello World!')
        'Executes a function with multiple return values.'

        def Func(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return (a + b, a - b)
        aval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
        bval = np.array([5, 6, 7, 8]).reshape([2, 2]).astype(np.float32)
        expected = Func(aval, bval)
        with self.session():

            @function.Defun(dtypes.float32, dtypes.float32)
            def Foo(a, b):
                if False:
                    return 10
                return Func(a, b)
            a = constant_op.constant(aval, name='a')
            b = constant_op.constant(bval, name='b')
            with self.test_scope():
                call_f = Foo(a, b)
            result = self.evaluate(call_f)
        self.assertAllClose(result, expected, rtol=0.001)

    def testCompileTimeConstantsInDefun(self):
        if False:
            while True:
                i = 10
        'Tests that XLA handles compile-time constants in defuns.'
        with self.session() as sess:

            @function.Defun(dtypes.float32, dtypes.int32, dtypes.int32)
            def Foo(a, c, d):
                if False:
                    return 10
                x = array_ops.slice(a, c, d)
                return x
            a = array_ops.placeholder(dtypes.float32)
            c = array_ops.placeholder(dtypes.int32, shape=[4])
            d = array_ops.placeholder(dtypes.int32, shape=[4])
            with self.test_scope():
                call_f = Foo(a, c, d)
            result = sess.run(call_f, feed_dict={a: np.ones([1, 4, 4, 1]), c: [0, 0, 0, 0], d: [1, 2, 2, 1]})
        self.assertAllEqual(np.ones([1, 2, 2, 1]), result)

    def DISABLED_testFunctionsNoInline(self):
        if False:
            return 10

        @function.Defun(dtypes.float32, noinline=True)
        def TimesTwo(x):
            if False:
                print('Hello World!')
            return x * 2

        @function.Defun(dtypes.float32, dtypes.float32)
        def APlus2B(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return a + TimesTwo(b)
        aval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
        bval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
        expected = aval + bval * 2
        with self.session() as sess:
            with self.test_scope():
                a = array_ops.placeholder(dtypes.float32, name='a')
                b = array_ops.placeholder(dtypes.float32, name='b')
                call = APlus2B(a, b)
            result = sess.run(call, {a: aval, b: bval})
        self.assertAllClose(result, expected, rtol=0.001)
if __name__ == '__main__':
    googletest.main()