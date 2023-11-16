"""Test cases for eager execution using XLA."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.layers import convolutional
from tensorflow.python.layers import pooling
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import googletest
from tensorflow.python.training import adam

class EagerTest(xla_test.XLATestCase):

    def testBasic(self):
        if False:
            for i in range(10):
                print('nop')
        with self.test_scope():
            three = constant_op.constant(3)
            five = constant_op.constant(5)
            product = three * five
            self.assertAllEqual(15, product)

    def testGradientTape(self):
        if False:
            while True:
                i = 10
        with self.test_scope():
            x = constant_op.constant(1.0)
            y = constant_op.constant(10.0)
            with backprop.GradientTape(persistent=True) as tape:
                tape.watch(x)
                tape.watch(y)
                a = x + y + x * y
            da_dx = tape.gradient(a, x)
            da_dy = tape.gradient(a, y)
        self.assertEqual(11.0, da_dx.numpy())
        self.assertEqual(2.0, da_dy.numpy())

    def testExecuteListOutputLen0(self):
        if False:
            return 10
        with self.test_scope():
            empty = constant_op.constant([], dtype=dtypes.float32)
            result = array_ops_stack.unstack(empty, 0)
            self.assertTrue(isinstance(result, list))
            self.assertEqual(0, len(result))

    def testExecuteListOutputLen1(self):
        if False:
            i = 10
            return i + 15
        with self.test_scope():
            split_dim = constant_op.constant(1)
            value = constant_op.constant([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
            result = array_ops.split(value, 1, axis=split_dim)
            self.assertTrue(isinstance(result, list))
            self.assertEqual(1, len(result))
            self.assertAllEqual([[0, 1, 2], [3, 4, 5]], result[0])

    def testExecuteListOutputLen3(self):
        if False:
            print('Hello World!')
        with self.test_scope():
            split_dim = constant_op.constant(1)
            value = constant_op.constant([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
            result = array_ops.split(value, 3, axis=split_dim)
            self.assertTrue(isinstance(result, list))
            self.assertEqual(3, len(result))
            self.assertAllEqual([[0], [3]], result[0])
            self.assertAllEqual([[1], [4]], result[1])
            self.assertAllEqual([[2], [5]], result[2])

    def testBasicGraph(self):
        if False:
            i = 10
            return i + 15
        with self.test_scope():
            three = constant_op.constant(3)
            five = constant_op.constant(5)
            product = three * five
            self.assertAllEqual(15, product)
        with context.graph_mode(), self.session():
            with self.test_scope():
                three = constant_op.constant(3)
                five = constant_op.constant(5)
                product = three * five
                self.assertAllEqual(15, self.evaluate(product))

    def testDegenerateSlices(self):
        if False:
            while True:
                i = 10
        with self.test_scope():
            npt = np.arange(1, 19, dtype=np.float32).reshape(3, 2, 3)
            t = constant_op.constant(npt)
            self.assertAllEqual(npt[0:-1:-1, :, :], t[0:-1:-1, :, :])
            self.assertAllEqual(npt[-1:0, :, :], t[-1:0, :, :])
            self.assertAllEqual(npt[-1:0, 2:2, 2:3:-1], t[-1:0, 2:2, 2:3:-1])

    def testIdentity(self):
        if False:
            return 10
        with self.test_scope():
            self.assertAllEqual(2, array_ops.identity(2))

    def testRandomOps(self):
        if False:
            while True:
                i = 10
        with self.test_scope():
            tensor = gen_random_ops.random_uniform((2, 2), dtypes.float32)
            row0 = tensor[0].numpy()
            row1 = tensor[1].numpy()
            self.assertFalse((row0 == row1).all())

    def testIdentityOnVariable(self):
        if False:
            for i in range(10):
                print('nop')
        with self.test_scope():
            v = resource_variable_ops.ResourceVariable(True)
            i = array_ops.identity(v)
        self.assertAllEqual(True, i.numpy())

    def testAssignAddVariable(self):
        if False:
            print('Hello World!')
        with self.test_scope():
            v = resource_variable_ops.ResourceVariable(1.0)
            v.assign_add(2.0)
        self.assertEqual(3.0, v.numpy())

    def testReadAssignRead(self):
        if False:
            i = 10
            return i + 15
        with self.test_scope():
            v = resource_variable_ops.ResourceVariable(1.0)
            val1 = v.read_value()
            v.assign_add(2.0)
            val2 = v.read_value()
        self.assertEqual(1.0, val1.numpy())
        self.assertEqual(3.0, val2.numpy())

    def testGradient(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                print('Hello World!')
            return x
        with self.test_scope():
            grad_fn = backprop.gradients_function(f)
            self.assertAllEqual(2.0, grad_fn(1.0, dy=2.0)[0])

    def testVariableGradient(self):
        if False:
            while True:
                i = 10
        with self.test_scope():
            v0 = resource_variable_ops.ResourceVariable(1.0)

            def f():
                if False:
                    return 10
                x = v0 * v0
                return x
            grads = backprop.implicit_grad(f)()
        self.assertEqual(2.0, grads[0][0].numpy())

    def testMultipleVariableReads(self):
        if False:
            return 10
        with self.test_scope():
            var = resource_variable_ops.ResourceVariable(array_ops.ones([32, 1024, 1024]))
            values = []
            for _ in range(100):
                values.append(var.value())

    def testShape(self):
        if False:
            print('Hello World!')

        def const(value):
            if False:
                for i in range(10):
                    print('nop')
            return array_ops.shape(constant_op.constant(value)).numpy()

        def ones(value):
            if False:
                i = 10
                return i + 15
            return array_ops.shape(array_ops.ones(value)).numpy()
        with self.test_scope():
            self.assertAllEqual([], const(3))
            self.assertAllEqual([3], const([1.0, 2.0, 3.0]))
            self.assertAllEqual([2, 2], const([[1.0, 2.0], [3.0, 4.0]]))
            self.assertAllEqual([2, 1, 2], const([[[1.0, 2.0]], [[3.0, 4.0]]]))
            self.assertAllEqual([], ones([]))
            self.assertAllEqual([3], ones([3]))
            self.assertAllEqual([2, 2], ones([2, 2]))
            self.assertAllEqual([2, 1, 2], ones([2, 1, 2]))

    def testShapeN(self):
        if False:
            return 10
        with self.test_scope():
            shapes = array_ops.shape_n([constant_op.constant(1.0), constant_op.constant([1.0, 2.0, 3.0]), constant_op.constant([[1.0, 2.0], [3.0, 4.0]])])
            self.assertAllEqual([[], [3], [2, 2]], [x.numpy().tolist() for x in shapes])
            shapes = array_ops.shape_n([array_ops.ones([]), array_ops.ones([3]), array_ops.ones([2, 2])])
            self.assertAllEqual([[], [3], [2, 2]], [x.numpy().tolist() for x in shapes])

    def testSize(self):
        if False:
            for i in range(10):
                print('nop')
        with self.test_scope():
            self.assertEqual(1, array_ops.size(constant_op.constant(1.0)).numpy())
            self.assertEqual(3, array_ops.size(constant_op.constant([1.0, 2.0, 3.0])).numpy())
            self.assertEqual(4, array_ops.size(constant_op.constant([[1.0, 2.0], [3.0, 4.0]])).numpy())

    def testRank(self):
        if False:
            i = 10
            return i + 15
        with self.test_scope():
            self.assertEqual(0, array_ops.rank(constant_op.constant(1.0)).numpy())
            self.assertEqual(1, array_ops.rank(constant_op.constant([1.0, 2.0, 3.0])).numpy())
            self.assertEqual(2, array_ops.rank(constant_op.constant([[1.0, 2.0], [3.0, 4.0]])).numpy())

    def testAdam(self):
        if False:
            print('Hello World!')
        with self.test_scope():
            optimizer = adam.AdamOptimizer(0.1)
            x = resource_variable_ops.ResourceVariable(10.0)
            with backprop.GradientTape() as tape:
                y = x * x
            dy_dx = tape.gradient(y, x)
            optimizer.apply_gradients([(dy_dx, x)])
            self.assertAlmostEqual(9.9, x.numpy(), places=3)

    def testAdamSparse(self):
        if False:
            print('Hello World!')
        with ops.device('/cpu:0'):
            embedding_matrix = resource_variable_ops.ResourceVariable(array_ops.ones([3, 2]))
        with self.test_scope():
            with backprop.GradientTape() as tape:
                embedding = embedding_ops.embedding_lookup(embedding_matrix, [1])
                y = math_ops.reduce_sum(embedding)
            dy_dx = tape.gradient(y, embedding_matrix)
            self.assertIsInstance(dy_dx, indexed_slices.IndexedSlices)
            optimizer = adam.AdamOptimizer(0.1)
            optimizer.apply_gradients([(dy_dx, embedding_matrix)])
            embedding_matrix.assign_add(array_ops.ones([3, 2]))
        self.assertAllClose([[2.0, 2.0], [1.9, 1.9], [2.0, 2.0]], embedding_matrix.numpy())

class EagerFunctionTest(xla_test.XLATestCase):

    def testBasic(self):
        if False:
            return 10
        with self.test_scope():
            matmul = def_function.function(math_ops.matmul)
            t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
            sq = matmul(t, t, transpose_a=True)
            self.assertAllEqual(sq.numpy().reshape(-1), [10, 14, 14, 20])

    def testConv(self):
        if False:
            return 10
        if 'GPU' in self.device:
            self.skipTest('Current implementation of RandomStandardNormal kernel is very slow on GPU, and has been denylisted.')
        with self.test_scope():
            data_format = 'channels_last'
            conv = convolutional.Conv2D(filters=1, kernel_size=2, padding='VALID', data_format=data_format, activation=nn_ops.relu, kernel_initializer=init_ops.ones_initializer(), bias_initializer=init_ops.zeros_initializer())
            pool = pooling.MaxPooling2D(2, 2, data_format=data_format)

            def model(x):
                if False:
                    return 10
                x = conv(x)
                return pool(x)
            model = def_function.function(model)
            x = array_ops.ones([1, 4, 4, 1])
            y = model(x)
            self.assertAllEqual(y.numpy(), [[[[4.0]]]])

    def testReadVariable(self):
        if False:
            for i in range(10):
                print('nop')
        with self.test_scope():
            v = resource_variable_ops.ResourceVariable(1.0)

            @def_function.function
            def f():
                if False:
                    i = 10
                    return i + 15
                return v.read_value()
            var = f()
            self.assertEqual(1.0, var.numpy())

    def testResourceVariableNoInlineReadWrite(self):
        if False:
            while True:
                i = 10
        with self.test_scope():
            v = resource_variable_ops.ResourceVariable(1.0)
            w = resource_variable_ops.ResourceVariable(0.0)

            @def_function.function(experimental_attributes={'_noinline': True})
            def g(x):
                if False:
                    print('Hello World!')
                w.assign(w.read_value() + x)
                return v.read_value() + x * w.read_value()

            @def_function.function(experimental_attributes={'_noinline': True})
            def f():
                if False:
                    return 10
                return g(1.0) + g(2.0) + g(3.0) + g(4.0) + g(5.0)
            self.assertEqual(145.0, f().numpy())
            self.assertEqual(15.0, w.read_value().numpy())

    def testResourceVariableNoInlineReadOnly(self):
        if False:
            return 10
        with self.test_scope():
            v = resource_variable_ops.ResourceVariable(10.0)

            @def_function.function(experimental_attributes={'_noinline': True})
            def g():
                if False:
                    print('Hello World!')
                return v.read_value()

            @def_function.function(experimental_attributes={'_noinline': True})
            def f():
                if False:
                    print('Hello World!')
                return g() + g() + g() + g() + g()
            self.assertEqual(50.0, f().numpy())

    def testResourceVariableNoInlineWriteOnly(self):
        if False:
            while True:
                i = 10
        with self.test_scope():
            v = resource_variable_ops.ResourceVariable(0.0)

            @def_function.function(experimental_attributes={'_noinline': True})
            def g(x):
                if False:
                    i = 10
                    return i + 15
                v.assign(x)

            @def_function.function(experimental_attributes={'_noinline': True})
            def f():
                if False:
                    while True:
                        i = 10
                g(1.0)
                g(2.0)
                g(3.0)
                g(4.0)
                g(5.0)
            f()
            self.assertEqual(5.0, v.read_value().numpy())

    def testUpdateVariable(self):
        if False:
            i = 10
            return i + 15
        with self.test_scope():
            v = resource_variable_ops.ResourceVariable(1.0)

            def f(v):
                if False:
                    for i in range(10):
                        print('nop')
                v.assign_add(1.0)
                return v
            f = def_function.function(f)
            var = f(v)
            self.assertEqual(2.0, var.numpy())

    def testReturnResourceHandle(self):
        if False:
            while True:
                i = 10
        with self.test_scope():
            v = resource_variable_ops.ResourceVariable([[1.0, 2.0], [3.0, 4.0]])

            def f(v):
                if False:
                    print('Hello World!')
                return v.handle
            f = def_function.function(f)
            handle = f(v)
            self.assertAllEqual(v.numpy(), resource_variable_ops.read_variable_op(handle, dtypes.float32).numpy())

    def testReturnMultipleResourceHandles(self):
        if False:
            i = 10
            return i + 15
        with self.test_scope():
            v1 = resource_variable_ops.ResourceVariable(1.25)
            v2 = resource_variable_ops.ResourceVariable(2.0)

            def f(v):
                if False:
                    for i in range(10):
                        print('nop')
                return (v.handle, 3.0 * v, v2.handle, v + v2)
            f = def_function.function(f)
            (v1_handle, v1_times_3, v2_handle, variable_sum) = f(v1)
            self.assertAllEqual(v1.numpy(), resource_variable_ops.read_variable_op(v1_handle, dtypes.float32).numpy())
            self.assertEqual(3.75, v1_times_3.numpy())
            self.assertAllEqual(v2.numpy(), resource_variable_ops.read_variable_op(v2_handle, dtypes.float32).numpy())
            self.assertEqual(3.25, variable_sum.numpy())

    def testAllArgumentKinds(self):
        if False:
            return 10
        'Test a complex function that takes different argument kinds.\n\n    tf2xla machinery that translates, compiles, and runs defuns\n    classifies arguments into: compile-time constants, regular tensors,\n    and resources. This test creates a function with a mix of all these\n    kinds. Moreover, the order of function arguments is intentionally mixed up.\n\n    This also tests the case when the same argument is a compile-time constant\n    as well as used in an operation that normally expects its inputs to be\n    in device memory - addition in this case.\n    '
        with self.test_scope():

            def foo(c1, r1, v1, c2, v2, r2):
                if False:
                    i = 10
                    return i + 15
                a = c1 + r1
                b = math_ops.cast(c2, dtypes.float32) + v2
                c = array_ops.slice(v1, c1, c2)
                d = r2 * v2
                return (a, b, c, d)
            foo = def_function.function(foo)
            c1 = [0, 0]
            c2 = array_ops.ones([2], dtype=dtypes.int32)
            r1 = array_ops.ones([2])
            r2 = [[2.0, 2.0], [3.0, 3.0]]
            v1 = resource_variable_ops.ResourceVariable([[1.0, 2.0], [3.0, 4.0]])
            v2 = resource_variable_ops.ResourceVariable([[10.0, 20.0], [30.0, 40.0]])
            (a, b, c, d) = foo(c1, r1, v1, c2, v2, r2)
            self.assertAllEqual([1, 1], a.numpy())
            self.assertAllEqual([[11.0, 21.0], [31.0, 41.0]], b.numpy())
            self.assertAllEqual([[1.0]], c.numpy())
            self.assertAllEqual([[20.0, 40.0], [90.0, 120.0]], d.numpy())

    def testDefunInGradientTape(self):
        if False:
            return 10
        with self.test_scope():
            v0 = resource_variable_ops.ResourceVariable(5.0)

            @def_function.function
            def f(x):
                if False:
                    i = 10
                    return i + 15
                x = v0 * v0 * x
                return x
            x = constant_op.constant(3.0)
            with backprop.GradientTape() as tape:
                y = f(x)
            dy = tape.gradient(y, v0)
        self.assertEqual(75, y.numpy())
        self.assertEqual(30, dy.numpy())

    def testGradientTapeInDefun(self):
        if False:
            print('Hello World!')
        with self.test_scope():
            v0 = resource_variable_ops.ResourceVariable(5.0)

            @def_function.function
            def f():
                if False:
                    for i in range(10):
                        print('nop')
                x = constant_op.constant(1.0)
                with backprop.GradientTape() as tape:
                    y = v0 * x
                dy = tape.gradient(y, v0)
                return dy
            dy = f()
            self.assertEqual(1.0, dy.numpy())

    def testSliceInDefun(self):
        if False:
            for i in range(10):
                print('nop')
        with self.test_scope():

            @def_function.function
            def f(x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return x[0::2, y:, ...]
            x = array_ops.ones([2, 3, 4], dtype=dtypes.float32)
            y = array_ops.ones([], dtype=dtypes.int32)
            with backprop.GradientTape() as tape:
                tape.watch(x)
                tape.watch(y)
                z = f(x, y)
            dz = tape.gradient(z, x)
            self.assertAllEqual(np.ones([1, 2, 4]), z.numpy())
            self.assertAllEqual((2, 3, 4), dz.shape.as_list())

    def testNestedDefun(self):
        if False:
            i = 10
            return i + 15
        with self.test_scope():

            @def_function.function
            def times_two(x):
                if False:
                    return 10
                return 2.0 * x

            @def_function.function
            def two_x_plus_1(x):
                if False:
                    print('Hello World!')
                return times_two(x) + 1.0
            x = constant_op.constant([2.0, 3.0, 4.0])
            y = two_x_plus_1(x)
            self.assertAllEqual([5.0, 7.0, 9.0], y.numpy())

    def testNestedDefunWithVariable(self):
        if False:
            for i in range(10):
                print('nop')
        with self.test_scope():
            v0 = resource_variable_ops.ResourceVariable(5.0)

            @def_function.function
            def g(x):
                if False:
                    return 10
                x = v0 * x
                return x

            @def_function.function
            def f(x):
                if False:
                    while True:
                        i = 10
                x = g(v0 * x)
                return x
            x = constant_op.constant(3.0)
            y = f(x)
        self.assertEqual(75.0, y.numpy())

    def testNestedDefunInGradientTape(self):
        if False:
            i = 10
            return i + 15
        with self.test_scope():
            v0 = resource_variable_ops.ResourceVariable(5.0)

            @def_function.function
            def g(x):
                if False:
                    for i in range(10):
                        print('nop')
                x = v0 * x
                return x

            @def_function.function
            def f(x):
                if False:
                    for i in range(10):
                        print('nop')
                x = g(v0 * x)
                return x
            x = constant_op.constant(3.0)
            with backprop.GradientTape() as tape:
                y = f(x)
            dy = tape.gradient(y, v0)
        self.assertEqual(75, y.numpy())
        self.assertEqual(30, dy.numpy())

    def testNestedDefunInGradientTapeDifferentVars(self):
        if False:
            while True:
                i = 10
        with self.test_scope():
            v0 = resource_variable_ops.ResourceVariable(5.0)
            v1 = resource_variable_ops.ResourceVariable(3.0)

            @def_function.function
            def g(x):
                if False:
                    print('Hello World!')
                x = v1 * x
                return x

            @def_function.function
            def f(x):
                if False:
                    i = 10
                    return i + 15
                x = g(v0 * x)
                return x
            x = constant_op.constant(3.0)
            with backprop.GradientTape(persistent=True) as tape:
                y = f(x)
            dy_v0 = tape.gradient(y, v0)
            dy_v1 = tape.gradient(y, v1)
        self.assertEqual(45, y.numpy())
        self.assertEqual(9, dy_v0.numpy())
        self.assertEqual(15, dy_v1.numpy())

    def testWhileInDefun(self):
        if False:
            i = 10
            return i + 15
        with self.test_scope():

            @def_function.function
            def f(start):
                if False:
                    return 10
                c = lambda x: math_ops.less(x, 13.0)
                b = lambda x: math_ops.add(x, 1.0)
                return while_loop.while_loop(c, b, [start])
            y = f(constant_op.constant(3.0))
        self.assertEqual(13.0, y.numpy())

    def testAutoGraphWhileInDefun(self):
        if False:
            while True:
                i = 10
        with self.test_scope():

            @def_function.function
            def f(start):
                if False:
                    i = 10
                    return i + 15
                x = start
                while x < 13.0:
                    x += 1.0
                return x
            y = f(constant_op.constant(3.0))
        self.assertEqual(13.0, y.numpy())

    def testCondInDefun(self):
        if False:
            while True:
                i = 10
        with self.test_scope():

            @def_function.function
            def f(pred, value):
                if False:
                    for i in range(10):
                        print('nop')
                fn1 = lambda : math_ops.add(value, 1.0)
                fn2 = lambda : math_ops.subtract(value, 1.0)
                return cond.cond(pred, fn1, fn2)
            plus_one = f(constant_op.constant(True), constant_op.constant(10.0))
            minus_one = f(constant_op.constant(False), constant_op.constant(10.0))
        self.assertEqual(11.0, plus_one.numpy())
        self.assertEqual(9.0, minus_one.numpy())

    def testAutoGraphCondInDefun(self):
        if False:
            return 10
        with self.test_scope():

            @def_function.function
            def f(pred, value):
                if False:
                    while True:
                        i = 10
                if pred:
                    return value + 1.0
                else:
                    return value - 1.0
            plus_one = f(constant_op.constant(True), constant_op.constant(10.0))
            minus_one = f(constant_op.constant(False), constant_op.constant(10.0))
        self.assertEqual(11.0, plus_one.numpy())
        self.assertEqual(9.0, minus_one.numpy())

    def testScanInDefun(self):
        if False:
            for i in range(10):
                print('nop')
        with self.test_scope():
            elems = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='data')
            v = constant_op.constant(2.0, name='v')

            @def_function.function
            def f(y):
                if False:
                    for i in range(10):
                        print('nop')
                return functional_ops.scan(lambda a, x: math_ops.multiply(a, x), y, initializer=v)
            r = f(elems)
            self.assertAllEqual([2.0, 4.0, 12.0, 48.0, 240.0, 1440.0], self.evaluate(r))

    def testFeedDeviceMemoryToOpExpectingHostMemory(self):
        if False:
            return 10

        @def_function.function
        def f(dims, value):
            if False:
                print('Hello World!')
            return array_ops.fill(dims, value)
        with self.test_scope():
            x = constant_op.constant([4], dtype=dtypes.int64)
        y = f(x, 3)
        self.assertAllEqual([3, 3, 3, 3], y)

    def testRequestNotToCompile(self):
        if False:
            print('Hello World!')
        with self.test_scope():

            def f(x):
                if False:
                    return 10
                with ops.device('device:CPU:0'):
                    y = 2.0 * x
                return (x, y)
            wholly_compiled_f = def_function.function(f)
            op_by_op_f = def_function.function(f, jit_compile=False)
            x = array_ops.identity([0.0, 2.0], name='data')
            (r_x, r_y) = wholly_compiled_f(x)
            self.assertAllEqual([0.0, 2.0], r_x)
            self.assertAllEqual([0.0, 4.0], r_y)
            if context.executing_eagerly():
                self.assertRegex(r_x.backing_device, self.device)
                self.assertRegex(r_y.backing_device, self.device)
            (r_x, r_y) = op_by_op_f(x)
            self.assertAllEqual([0.0, 2.0], r_x)
            self.assertAllEqual([0.0, 4.0], r_y)
            if context.executing_eagerly():
                self.assertRegex(r_x.backing_device, self.device)
                self.assertRegex(r_y.backing_device, 'device:CPU:0')

class ExcessivePaddingTest(xla_test.XLATestCase):
    """Test that eager execution works with TPU flattened tensors.

  Tensors that would normally be excessively padded when written
  to TPU memory are reshaped to 1-D flat tensors.

  This test case verifies that such tensors work with eager execution.

  The flattening currently only happens on TPU, but tests should work
  fine with all backends as flattening is transparent.
  """

    def testFromConstant(self):
        if False:
            for i in range(10):
                print('nop')
        with self.test_scope():
            tensor = constant_op.constant(100 * [[[10.0], [2.0]]])
            reduced = math_ops.reduce_sum(tensor, axis=1)
            self.assertAllEqual(100 * [[12.0]], reduced)

    def testFromOperation(self):
        if False:
            for i in range(10):
                print('nop')
        with self.test_scope():
            tensor = array_ops.ones([3, 100, 2, 2])
            reduced = math_ops.reduce_sum(tensor, axis=[0, 2, 3])
            self.assertAllEqual(100 * [12.0], reduced)

    def testAsFunctionInput(self):
        if False:
            return 10
        with self.test_scope():

            @def_function.function
            def f(x):
                if False:
                    for i in range(10):
                        print('nop')
                return math_ops.reduce_sum(x, axis=2)
            tensor = constant_op.constant(100 * [[[10.0, 2.0]]])
            reduced = f(tensor)
            self.assertAllEqual(100 * [[12.0]], reduced)

    def testAsFunctionOutput(self):
        if False:
            print('Hello World!')
        with self.test_scope():

            @def_function.function
            def f(x):
                if False:
                    return 10
                return x * constant_op.constant(100 * [[[10.0, 2.0]]])
            y = f(3)
            reduced = math_ops.reduce_sum(y, axis=2)
            self.assertAllEqual(100 * [[36.0]], reduced)

def multiple_tpus():
    if False:
        while True:
            i = 10
    devices = context.context().devices()
    return len([d for d in devices if 'device:TPU:' in d]) > 1

class MultiDeviceTest(xla_test.XLATestCase):
    """Test running TPU computation on more than one core."""

    def testBasic(self):
        if False:
            return 10
        if not multiple_tpus():
            self.skipTest('MultiDeviceTest requires multiple TPU devices.')
        with ops.device('device:TPU:0'):
            two = constant_op.constant(2)
            five = constant_op.constant(5)
            ten = two * five
            self.assertAllEqual(10, ten)
        with ops.device('device:TPU:1'):
            two = constant_op.constant(2)
            three = constant_op.constant(3)
            six = two * three
            self.assertAllEqual(6, six)
        self.assertAllEqual(16, ten + six)
if __name__ == '__main__':
    ops.enable_eager_execution(config=config_pb2.ConfigProto(log_device_placement=True))
    googletest.main()