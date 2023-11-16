"""Tests for memory leaks in eager execution.

It is possible that this test suite will eventually become flaky due to taking
too long to run (since the tests iterate many times), but for now they are
helpful for finding memory leaks since not all PyObject leaks are found by
introspection (test_util decorators). Please be careful adding new tests here.
"""
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.eager.memory_tests import memory_test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients as gradient_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.variables import Variable

class MemoryTest(test.TestCase):

    def testMemoryLeakAnonymousVariable(self):
        if False:
            return 10
        if not memory_test_util.memory_profiler_is_available():
            self.skipTest('memory_profiler required to run this test')

        def f():
            if False:
                while True:
                    i = 10
            inputs = Variable(array_ops.zeros([32, 100], dtypes.float32))
            del inputs
        memory_test_util.assert_no_leak(f, num_iters=10000, increase_threshold_absolute_mb=10)

    def testMemoryLeakInFunction(self):
        if False:
            return 10
        if not memory_test_util.memory_profiler_is_available():
            self.skipTest('memory_profiler required to run this test')

        def f():
            if False:
                print('Hello World!')

            @def_function.function
            def graph(x):
                if False:
                    print('Hello World!')
                return x * x + x
            graph(constant_op.constant(42))
        memory_test_util.assert_no_leak(f, num_iters=1000, increase_threshold_absolute_mb=30)

    @test_util.assert_no_new_pyobjects_executing_eagerly
    def testNestedFunctionsDeleted(self):
        if False:
            while True:
                i = 10

        @def_function.function
        def f(x):
            if False:
                return 10

            @def_function.function
            def my_sin(x):
                if False:
                    while True:
                        i = 10
                return math_ops.sin(x)
            return my_sin(x)
        x = constant_op.constant(1.0)
        with backprop.GradientTape() as t1:
            t1.watch(x)
            with backprop.GradientTape() as t2:
                t2.watch(x)
                y = f(x)
            dy_dx = t2.gradient(y, x)
        dy2_dx2 = t1.gradient(dy_dx, x)
        self.assertAllClose(0.84147096, y.numpy())
        self.assertAllClose(0.5403023, dy_dx.numpy())
        self.assertAllClose(-0.84147096, dy2_dx2.numpy())

    def testMemoryLeakInGlobalGradientRegistry(self):
        if False:
            for i in range(10):
                print('nop')
        if not memory_test_util.memory_profiler_is_available():
            self.skipTest('memory_profiler required to run this test')

        def f():
            if False:
                return 10

            @def_function.function(autograph=False)
            def graph(x):
                if False:
                    for i in range(10):
                        print('nop')

                @def_function.function(autograph=False)
                def cubed(a):
                    if False:
                        i = 10
                        return i + 15
                    return a * a * a
                y = cubed(x)
                del cubed
                return gradient_ops.gradients(gradient_ops.gradients(y, x), x)
            return graph(constant_op.constant(1.5))[0].numpy()
        memory_test_util.assert_no_leak(f, num_iters=300, increase_threshold_absolute_mb=50)
if __name__ == '__main__':
    test.main()