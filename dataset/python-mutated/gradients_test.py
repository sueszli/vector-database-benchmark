from absl.testing import parameterized
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
_COS_DERIVATIVES = [math_ops.cos, lambda x: -math_ops.sin(x), lambda x: -math_ops.cos(x), math_ops.sin, math_ops.cos]

class FunctionGradientsTest(test.TestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(FunctionGradientsTest, self).setUp()
        cpus = config.list_physical_devices('CPU')
        config.set_logical_device_configuration(cpus[0], [context.LogicalDeviceConfiguration(), context.LogicalDeviceConfiguration(), context.LogicalDeviceConfiguration(), context.LogicalDeviceConfiguration()])

    def testGraphModeWithGradients(self):
        if False:
            for i in range(10):
                print('nop')
        v = resource_variable_ops.ResourceVariable(1.0, name='v')

        @polymorphic_function.function
        def step():
            if False:
                while True:
                    i = 10

            def inner():
                if False:
                    while True:
                        i = 10
                return v * v
            return backprop.implicit_grad(inner)()[0][0]
        self.assertAllEqual(step(), 2.0)

    def testGraphGradientVariable(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default(), self.cached_session():
            v = variables.Variable(1.0)

            @polymorphic_function.function
            def f():
                if False:
                    print('Hello World!')
                return 2.0 * v
            node = f()
            (grads,) = gradients_impl.gradients(node, v)
            v.initializer.run()
            self.assertAllEqual(grads, 2.0)
            self.assertEqual(grads.shape, v.shape)

    def testSymbolicHigherOrder(self):
        if False:
            while True:
                i = 10

        @polymorphic_function.function
        def f(x, order):
            if False:
                return 10
            y = polymorphic_function.function(lambda : math_ops.cos(x))()
            for _ in range(order):
                (y,) = gradients_impl.gradients(y, [x])
            return y
        for (order, expected) in enumerate(_COS_DERIVATIVES):
            self.assertAllClose(expected(constant_op.constant(1.0)), f(constant_op.constant(1.0), order))

    @parameterized.parameters([dict(persistent=True), dict(persistent=False)])
    def testSymbolicHigherOrderUnderTape(self, persistent):
        if False:
            return 10

        @polymorphic_function.function
        def f(x, order):
            if False:
                return 10
            with backprop.GradientTape(persistent=persistent) as tape:
                tape.watch(x)
                y = polymorphic_function.function(lambda : math_ops.cos(x))()
            tape_dy = tape.gradient(y, x)
            for _ in range(order):
                (y,) = gradients_impl.gradients(y, [x])
            if order > 0:
                y1 = tape_dy
                for _ in range(order - 1):
                    (y1,) = gradients_impl.gradients(y1, [x])
            else:
                y1 = y
            return (y, y1)
        for (order, expected_f) in enumerate(_COS_DERIVATIVES):
            expected = self.evaluate(expected_f(constant_op.constant(1.0)))
            self.assertAllClose((expected, expected), f(constant_op.constant(1.0), order))

    def testIteratedGradientsNested(self):
        if False:
            return 10

        def _grad(f):
            if False:
                for i in range(10):
                    print('nop')

            def _grad_function(primal):
                if False:
                    print('Hello World!')
                with backprop.GradientTape() as tape:
                    tape.watch(primal)
                    primal_out = f(primal)
                return tape.gradient(primal_out, primal)
            return _grad_function

        @polymorphic_function.function
        def _forward(x):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.cos(x)
        f = _forward
        traced_f = polymorphic_function.function(f)
        one = constant_op.constant(1.0)
        for expected in _COS_DERIVATIVES:
            self.assertAllClose(expected(one), f(one))
            self.assertAllClose(expected(one), traced_f(one))
            self.assertAllClose(expected(one), polymorphic_function.function(f)(one))
            f = _grad(f)
            traced_f = polymorphic_function.function(_grad(traced_f))

    def testIteratedGradientsNestedWithVariable(self):
        if False:
            for i in range(10):
                print('nop')

        def _grad(f):
            if False:
                return 10

            def _grad_function():
                if False:
                    print('Hello World!')
                with backprop.GradientTape() as tape:
                    primal_out = f()
                (g,) = tape.gradient(primal_out, tape.watched_variables())
                return g
            return _grad_function
        v = variables.Variable(2.0)

        @polymorphic_function.function
        def _forward():
            if False:
                print('Hello World!')
            return math_ops.cos(v)
        f = _forward
        two = constant_op.constant(2.0)
        for expected in _COS_DERIVATIVES:
            self.assertAllClose(expected(two), f())
            self.assertAllClose(expected(two), polymorphic_function.function(f)())
            f = _grad(f)

    def testIteratedGradientsPersistent(self):
        if False:
            i = 10
            return i + 15

        @polymorphic_function.function
        def _forward(z):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.cos(z)
        f = _forward
        with backprop.GradientTape(persistent=True) as tape:
            start = constant_op.constant(1.0)
            tape.watch(start)
            x = f(start)
            for expected in _COS_DERIVATIVES:
                self.assertAllClose(expected(start), x)
                x = tape.gradient(x, start)

    def testHigherOrderWithVariable(self):
        if False:
            i = 10
            return i + 15
        v = variables.Variable(1.0)

        @polymorphic_function.function
        def _forward():
            if False:
                i = 10
                return i + 15
            return math_ops.cos(v)
        f = _forward
        with backprop.GradientTape(persistent=True) as tape:
            x = f()
            for expected in _COS_DERIVATIVES:
                self.assertAllClose(expected(constant_op.constant(1.0)), x)
                (x,) = tape.gradient(x, tape.watched_variables())

    def testGradientsChained(self):
        if False:
            for i in range(10):
                print('nop')

        @polymorphic_function.function
        def _forward(z):
            if False:
                i = 10
                return i + 15
            return math_ops.cos(z)
        f = _forward
        x = constant_op.constant(1.0)
        with backprop.GradientTape() as t:
            t.watch(x)
            y = f(x)
        with backprop.GradientTape() as tt:
            doutputs = constant_op.constant(2.0)
            tt.watch(doutputs)
            g = t.gradient(y, x, doutputs)
        self.assertAllClose(-2.0 * math_ops.sin(x), g)
        gg = tt.gradient(g, doutputs)
        self.assertAllClose(-math_ops.sin(x), gg)

    def testSymGradGatherNd(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default(), self.cached_session():

            @polymorphic_function.function
            def f(x):
                if False:
                    i = 10
                    return i + 15
                return array_ops.gather_nd(x, [[0]])
            c = constant_op.constant([[2.0]])
            f_c = f(c)
            (g,) = gradients_impl.gradients(f_c, c)
            self.assertAllEqual(self.evaluate(g).values, [[1.0]])

    def testNoSymGradNestedDefun(self):
        if False:
            i = 10
            return i + 15

        @polymorphic_function.function
        def outer():
            if False:
                for i in range(10):
                    print('nop')

            @polymorphic_function.function
            def f(x):
                if False:
                    return 10
                return array_ops.gather_nd(x, [[0]])
            c = constant_op.constant([[2.0]])
            f_c = f(c)
            (g,) = gradients_impl.gradients(f_c, c)
            self.assertIsInstance(g, indexed_slices.IndexedSlices)
        outer()

    def testGraphFunctionWithGradients(self):
        if False:
            i = 10
            return i + 15
        v = resource_variable_ops.ResourceVariable(1.0, name='v')

        @polymorphic_function.function
        def step():
            if False:
                i = 10
                return i + 15

            def inner():
                if False:
                    return 10
                return v * v
            return backprop.implicit_grad(inner)()[0][0]
        step_op = step.get_concrete_function()
        self.assertEqual(step_op.output_dtypes, dtypes.float32)
        self.assertEqual(step_op.output_shapes, tensor_shape.TensorShape([]))
        self.assertAllEqual(step_op(), 2.0)

    @test_util.run_in_graph_and_eager_modes()
    def testDefunCondGradient(self):
        if False:
            for i in range(10):
                print('nop')

        @polymorphic_function.function
        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return cond.cond(x > 0.5, lambda : 2 * x, lambda : 3 * x)
        with backprop.GradientTape() as t:
            x = constant_op.constant(1.0)
            t.watch(x)
            y = f(x)
        self.assertAllEqual(self.evaluate(t.gradient(y, x)), 2.0)

    @test_util.run_in_graph_and_eager_modes()
    def testGraphLoopGradient(self):
        if False:
            i = 10
            return i + 15

        @polymorphic_function.function
        def f(x):
            if False:
                return 10
            return while_loop.while_loop(lambda _, i: i < 2, lambda x, i: (2 * x, i + 1), [x, 0])[0]
        with backprop.GradientTape() as t:
            x = constant_op.constant(1.0)
            t.watch(x)
            y = f(x)
        self.assertAllEqual(self.evaluate(t.gradient(y, x)), 4.0)

    def testGraphLoopGradientInsideSession(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            n = constant_op.constant(2.0)
            x = array_ops.placeholder(dtypes.float32, shape=None)

            @polymorphic_function.function
            def f():
                if False:
                    while True:
                        i = 10
                c = lambda n: n < 10
                b = lambda n: n * x
                return while_loop.while_loop(c, b, [n], [tensor_shape.unknown_shape()])
            l = f()
            dx = gradients_impl.gradients(l, [x])[0]
            with self.cached_session():
                self.assertEqual(dx.eval(feed_dict={x: 2.0}), 24.0)

    def testDefunDifferentiable(self):
        if False:
            while True:
                i = 10
        v = resource_variable_ops.ResourceVariable(1.0)

        @polymorphic_function.function
        def f():
            if False:
                for i in range(10):
                    print('nop')
            return v * v
        self.assertAllEqual(backprop.implicit_grad(f)()[0][0], 2.0)

    def testDefunCanBeDifferentiatedTwice(self):
        if False:
            while True:
                i = 10
        v = resource_variable_ops.ResourceVariable(1.0)

        @polymorphic_function.function
        def f():
            if False:
                i = 10
                return i + 15
            return v * v
        self.assertAllEqual(backprop.implicit_grad(f)()[0][0], 2.0)
        self.assertAllEqual(backprop.implicit_grad(f)()[0][0], 2.0)

    def testSymbolicGradientVariableNoneNotZerosLike(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            v = variables.Variable(1.0)

            @polymorphic_function.function
            def f(x, v):
                if False:
                    i = 10
                    return i + 15
                v.read_value()
                return x * x
            x = constant_op.constant(1.0)
            l = f(x, v)
            (_, dv) = gradients_impl.gradients(l, [x, v])
            with self.cached_session():
                v.initializer.run()
                self.assertEqual(dv, None)

    def testDefunCallBackprop(self):
        if False:
            print('Hello World!')

        @polymorphic_function.function
        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.add(x, x)

        @polymorphic_function.function
        def g(x):
            if False:
                for i in range(10):
                    print('nop')
            return backprop.gradients_function(f, [0])(x)[0]
        self.assertAllEqual(2, g(constant_op.constant(2.0)))

    @test_util.run_v1_only('b/120545219')
    def testGraphModeEagerGradError(self):
        if False:
            i = 10
            return i + 15
        with context.graph_mode():

            def f():
                if False:
                    i = 10
                    return i + 15
                x = variable_scope.get_variable('v', initializer=constant_op.constant(1.0))
                return x * constant_op.constant(2.0)
            with self.assertRaisesRegex(ValueError, 'No trainable variables were accessed'):
                backprop.implicit_val_and_grad(f)()

    def testDefunCallBackpropUsingSameObjectForMultipleArguments(self):
        if False:
            return 10

        @polymorphic_function.function
        def g(x):
            if False:
                for i in range(10):
                    print('nop')
            return backprop.gradients_function(math_ops.multiply, [0, 1])(x, x)

        def np_g(x):
            if False:
                for i in range(10):
                    print('nop')
            return [d.numpy() for d in g(x)]
        x = constant_op.constant(1.0)
        self.assertAllEqual([1.0, 1.0], np_g(x))
        self.assertAllEqual([1.0, 1.0], np_g(1.0))

    def testGradientTensorConversionWithDefun(self):
        if False:
            while True:
                i = 10
        three = resource_variable_ops.ResourceVariable(3.0, name='v')

        @polymorphic_function.function
        def f(x):
            if False:
                return 10
            return math_ops.add(x, three)

        def g(x):
            if False:
                while True:
                    i = 10
            return f(x)
        g = backprop.implicit_grad(g)(constant_op.constant(1.0))[0][0]
        self.assertAllEqual(g, 1.0)

    def testGradient(self):
        if False:
            for i in range(10):
                print('nop')
        matmul = polymorphic_function.function(math_ops.matmul)

        def sq(x):
            if False:
                return 10
            return matmul(x, x, transpose_a=True)
        t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        (grad_t,) = backprop.gradients_function(sq, [0])(t)
        self.assertAllEqual(grad_t, [[6, 6], [14, 14]])

    def testGradientInFunction(self):
        if False:
            while True:
                i = 10

        @polymorphic_function.function
        def f(x):
            if False:
                i = 10
                return i + 15
            return backprop.gradients_function(lambda y: y * y, [0])(x)[0]
        self.assertAllEqual(f(constant_op.constant(1.0)), 2.0)

    def testGradientOfGatherWithDefun(self):
        if False:
            print('Hello World!')
        v = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0])

        def sum_gather():
            if False:
                print('Hello World!')
            return math_ops.reduce_sum(array_ops.gather(v, [1, 2]))
        grad_fn = backprop.implicit_grad(sum_gather)
        gradient = grad_fn()
        defun_grad_fn = backprop.implicit_grad(polymorphic_function.function(sum_gather))
        defun_gradient = defun_grad_fn()
        self.assertEqual(len(gradient), len(defun_gradient))
        gradient = gradient[0][0]
        defun_gradient = defun_gradient[0][0]
        self.assertAllEqual(gradient.values, defun_gradient.values)
        self.assertAllEqual(gradient.indices, defun_gradient.indices)
        self.assertAllEqual(gradient.dense_shape, defun_gradient.dense_shape)

    def testDifferentiableFunctionNoneOutputs(self):
        if False:
            while True:
                i = 10

        @polymorphic_function.function
        def my_function(x):
            if False:
                for i in range(10):
                    print('nop')
            return (x, None)

        def wrapper(x):
            if False:
                i = 10
                return i + 15
            return my_function(x)[0]
        g = backprop.gradients_function(wrapper, [0])(constant_op.constant(0.0))
        self.assertAllEqual(g[0], 1.0)

        @polymorphic_function.function
        def foo(a):
            if False:
                i = 10
                return i + 15
            return (None, a * a)
        x = constant_op.constant(5.0)
        with backprop.GradientTape() as tp:
            tp.watch(x)
            (none, r) = foo(x)
        g = tp.gradient(r, x)
        self.assertIs(none, None)
        self.assertAllEqual(r, 25.0)
        self.assertAllEqual(g, 2 * 5.0)

    @test_util.run_in_graph_and_eager_modes
    def testNestedDifferentiableFunction(self):
        if False:
            i = 10
            return i + 15

        @polymorphic_function.function
        def inner_fn(a, b):
            if False:
                i = 10
                return i + 15
            return a * math_ops.add(a, b)

        @polymorphic_function.function
        def outer_fn(x):
            if False:
                print('Hello World!')
            return inner_fn(x, 1.0)
        x = constant_op.constant(5.0)
        with backprop.GradientTape() as tp:
            tp.watch(x)
            result = outer_fn(x)
        grad = tp.gradient(result, x)
        self.assertAllEqual(grad, 2 * 5.0 + 1.0)

    @test_util.run_in_graph_and_eager_modes
    def testDeeplyNestedDifferentiableFunction(self):
        if False:
            while True:
                i = 10

        @polymorphic_function.function
        def inner_inner_fn(a, b):
            if False:
                return 10
            return math_ops.add(a, b)

        @polymorphic_function.function
        def inner_fn(a, b):
            if False:
                print('Hello World!')
            return inner_inner_fn(a, b)

        @polymorphic_function.function
        def middle_fn(a, b):
            if False:
                return 10
            return a * inner_fn(a, b)

        @polymorphic_function.function
        def outer_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return middle_fn(x, 1.0)
        x = constant_op.constant(5.0)
        with backprop.GradientTape() as tp:
            tp.watch(x)
            result = outer_fn(x)
        grad = tp.gradient(result, x)
        self.assertAllEqual(grad, 2 * 5.0 + 1.0)

    @test_util.run_in_graph_and_eager_modes
    def testDeeplyNestedDifferentiableFunctionWithMultipleGradCalls(self):
        if False:
            return 10

        @polymorphic_function.function
        def inner_fn(a, b):
            if False:
                return 10
            return math_ops.add(a, b)

        @polymorphic_function.function
        def middle_fn(a, b):
            if False:
                i = 10
                return i + 15
            return math_ops.mul(a, inner_fn(a, b))

        @polymorphic_function.function
        def outer_fn(x):
            if False:
                while True:
                    i = 10
            return middle_fn(x, 3.0)
        x = constant_op.constant(5.0)
        self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))
        with backprop.GradientTape() as tp:
            tp.watch(x)
            result = outer_fn(x)
        grad = tp.gradient(result, x)
        self.assertAllEqual(grad, 2 * 5.0 + 3.0)
        self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))
        self.assertAllEqual(middle_fn(3.0, x), 3.0 * (3.0 + 5.0))
        with backprop.GradientTape() as tp:
            tp.watch(x)
            result = outer_fn(x)
        grad = tp.gradient(result, x)
        self.assertAllEqual(grad, 2 * 5.0 + 3.0)
        y = constant_op.constant(4.0)
        with backprop.GradientTape() as tp:
            tp.watch(y)
            result = outer_fn(y)
        grad = tp.gradient(result, y)
        self.assertAllEqual(grad, 2 * 4.0 + 3.0)
        with backprop.GradientTape() as tp:
            tp.watch(y)
            result = inner_fn(y, y)
        grad = tp.gradient(result, y)
        self.assertAllEqual(grad, 2.0)

    @test_util.run_in_graph_and_eager_modes
    def testDeeplyNestedDifferentiableFunctionGradientTapeInDefun(self):
        if False:
            for i in range(10):
                print('nop')

        @polymorphic_function.function
        def inner_inner_fn(a, b):
            if False:
                return 10
            return math_ops.add(a, b)

        @polymorphic_function.function
        def inner_fn(a, b):
            if False:
                while True:
                    i = 10
            return inner_inner_fn(a, b)

        @polymorphic_function.function
        def middle_fn(a, b):
            if False:
                i = 10
                return i + 15
            return a * inner_fn(a, b)

        @polymorphic_function.function
        def outer_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            with backprop.GradientTape() as tp:
                tp.watch(x)
                result = middle_fn(x, 1.0)
            grad = tp.gradient(result, x)
            return grad
        x = constant_op.constant(5.0)
        grad = outer_fn(x)
        self.assertAllEqual(grad, 2 * 5.0 + 1.0)

    @test_util.run_in_graph_and_eager_modes
    def testDeeplyNestedDifferentiableFunctionGradientTapeInNestedDefun(self):
        if False:
            while True:
                i = 10

        @polymorphic_function.function
        def inner_inner_fn(a, b):
            if False:
                return 10
            return math_ops.add(a, b)

        @polymorphic_function.function
        def inner_fn(a, b):
            if False:
                i = 10
                return i + 15
            return inner_inner_fn(a, b)

        @polymorphic_function.function
        def middle_fn(a, b):
            if False:
                while True:
                    i = 10
            return a * inner_fn(a, b)

        @polymorphic_function.function
        def almost_outer_fn(x):
            if False:
                return 10
            with backprop.GradientTape() as tp:
                tp.watch(x)
                result = middle_fn(x, 1.0)
            grad = tp.gradient(result, x)
            return grad

        @polymorphic_function.function
        def outer_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return almost_outer_fn(x)
        x = constant_op.constant(5.0)
        grad = outer_fn(x)
        self.assertAllEqual(grad, 2 * 5.0 + 1.0)

    @test_util.run_in_graph_and_eager_modes
    def testDeeplyNestedDifferentiableFunctionGradientTapeInMultNestedDefun(self):
        if False:
            while True:
                i = 10

        @polymorphic_function.function
        def inner_inner_fn(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.add(a, b)

        @polymorphic_function.function
        def inner_fn(a, b):
            if False:
                print('Hello World!')
            return inner_inner_fn(a, b)

        @polymorphic_function.function
        def middle_fn(a, b):
            if False:
                print('Hello World!')
            return a * inner_fn(a, b)

        @polymorphic_function.function
        def almost_outer_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            with backprop.GradientTape() as tp:
                tp.watch(x)
                result = middle_fn(x, 1.0)
            grad = tp.gradient(result, x)
            return grad

        @polymorphic_function.function
        def outer_fn(x):
            if False:
                print('Hello World!')
            return almost_outer_fn(x)

        @polymorphic_function.function
        def outer_outer_fn(x):
            if False:
                while True:
                    i = 10
            return outer_fn(x)
        x = constant_op.constant(5.0)
        grad = outer_outer_fn(x)
        self.assertAllEqual(grad, 2 * 5.0 + 1.0)

    @test_util.run_in_graph_and_eager_modes
    def testDeeplyNestedDifferentiableFunctionTFGradientInDefun(self):
        if False:
            i = 10
            return i + 15

        @polymorphic_function.function
        def inner_inner_fn(a, b):
            if False:
                return 10
            return math_ops.add(a, b)

        @polymorphic_function.function
        def inner_fn(a, b):
            if False:
                return 10
            return inner_inner_fn(a, b)

        @polymorphic_function.function
        def middle_fn(a, b):
            if False:
                return 10
            return a * inner_fn(a, b)

        @polymorphic_function.function
        def outer_fn(x):
            if False:
                while True:
                    i = 10
            result = middle_fn(x, 1.0)
            return gradients_impl.gradients(result, [x])[0]
        x = constant_op.constant(5.0)
        grad = outer_fn(x)
        self.assertAllEqual(grad, 2 * 5.0 + 1.0)

    @test_util.run_in_graph_and_eager_modes
    def testDeeplyNestedDifferentiableFunctionTFGradientInNestedDefun(self):
        if False:
            for i in range(10):
                print('nop')

        @polymorphic_function.function
        def inner_inner_fn(a, b):
            if False:
                i = 10
                return i + 15
            return math_ops.add(a, b)

        @polymorphic_function.function
        def inner_fn(a, b):
            if False:
                i = 10
                return i + 15
            return inner_inner_fn(a, b)

        @polymorphic_function.function
        def middle_fn(a, b):
            if False:
                print('Hello World!')
            return a * inner_fn(a, b)

        @polymorphic_function.function
        def almost_outer_fn(x):
            if False:
                i = 10
                return i + 15
            result = middle_fn(x, 1.0)
            return gradients_impl.gradients(result, [x])[0]

        @polymorphic_function.function
        def outer_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return almost_outer_fn(x)
        x = constant_op.constant(5.0)
        grad = outer_fn(x)
        self.assertAllEqual(grad, 2 * 5.0 + 1.0)

    @test_util.run_in_graph_and_eager_modes
    def testDeeplyNestedDifferentiableFunctionTFGradientInMultNestedDefun(self):
        if False:
            print('Hello World!')

        @polymorphic_function.function
        def inner_inner_fn(a, b):
            if False:
                while True:
                    i = 10
            return math_ops.add(a, b)

        @polymorphic_function.function
        def inner_fn(a, b):
            if False:
                print('Hello World!')
            return inner_inner_fn(a, b)

        @polymorphic_function.function
        def middle_fn(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return a * inner_fn(a, b)

        @polymorphic_function.function
        def almost_outer_fn(x):
            if False:
                return 10
            result = middle_fn(x, 1.0)
            return gradients_impl.gradients(result, [x])[0]

        @polymorphic_function.function
        def outer_fn(x):
            if False:
                i = 10
                return i + 15
            return almost_outer_fn(x)

        @polymorphic_function.function
        def outer_outer_fn(x):
            if False:
                while True:
                    i = 10
            return outer_fn(x)
        x = constant_op.constant(5.0)
        grad = outer_outer_fn(x)
        self.assertAllEqual(grad, 2 * 5.0 + 1.0)

    def testDeeplyNestedDifferentiableFunctionWithVariable(self):
        if False:
            for i in range(10):
                print('nop')
        var = variables.Variable(constant_op.constant(1.0))

        @polymorphic_function.function
        def inner_fn(a, b):
            if False:
                return 10
            return math_ops.add(a, b)

        @polymorphic_function.function
        def middle_fn(a, b):
            if False:
                print('Hello World!')
            return a * inner_fn(a, b)

        @polymorphic_function.function
        def outer_fn(x):
            if False:
                i = 10
                return i + 15
            return middle_fn(x, var)
        x = constant_op.constant(5.0)
        with backprop.GradientTape() as tp:
            tp.watch(x)
            result = outer_fn(x)
        grad = tp.gradient(result, x)
        self.assertAllEqual(grad, 2 * 5.0 + 1.0)

    def testDeeplyNestedDifferentiableFunctionWithVariableMultipleGradCalls(self):
        if False:
            return 10
        v = variables.Variable(constant_op.constant(3.0))

        @polymorphic_function.function
        def inner_fn(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.add(a, b)

        @polymorphic_function.function
        def middle_fn(a, b):
            if False:
                print('Hello World!')
            return math_ops.mul(a, inner_fn(a, b))

        @polymorphic_function.function
        def outer_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return middle_fn(x, v)
        x = constant_op.constant(5.0)
        self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))
        with backprop.GradientTape() as tp:
            tp.watch(x)
            result = outer_fn(x)
        grad = tp.gradient(result, x)
        self.assertAllEqual(grad, 2 * 5.0 + 3.0)
        self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))
        self.assertAllEqual(middle_fn(v, x), 3.0 * (3.0 + 5.0))
        with backprop.GradientTape() as tp:
            tp.watch(x)
            result = outer_fn(x)
        grad = tp.gradient(result, x)
        self.assertAllEqual(grad, 2 * 5.0 + 3.0)
        y = constant_op.constant(4.0)
        with backprop.GradientTape() as tp:
            tp.watch(y)
            result = outer_fn(y)
        grad = tp.gradient(result, y)
        self.assertAllEqual(grad, 2 * 4.0 + 3.0)
        v.assign(constant_op.constant(1.5))
        with backprop.GradientTape() as tp:
            tp.watch(y)
            result = outer_fn(y)
        grad = tp.gradient(result, y)
        self.assertAllEqual(grad, 2 * 4.0 + 1.5)
        with backprop.GradientTape() as tp:
            tp.watch(y)
            result = inner_fn(y, v)
        grad = tp.gradient(result, y)
        self.assertAllEqual(grad, 1.0)

    def testDeeplyNestedDifferentiableFunctionWithVariableMultipleTFGrads(self):
        if False:
            i = 10
            return i + 15
        with context.graph_mode(), self.cached_session():
            v = resource_variable_ops.ResourceVariable(3.0)
            v.initializer.run()

            @polymorphic_function.function
            def inner_fn(a, b):
                if False:
                    while True:
                        i = 10
                return math_ops.add(a, b)

            @polymorphic_function.function
            def middle_fn(a, b):
                if False:
                    return 10
                return math_ops.mul(a, inner_fn(a, b))

            @polymorphic_function.function
            def outer_fn(x):
                if False:
                    for i in range(10):
                        print('nop')
                return middle_fn(x, v)
            x = constant_op.constant(5.0)
            self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))
            (grad,) = gradients_impl.gradients(outer_fn(x), x)
            self.assertAllEqual(grad, 2 * 5.0 + 3.0)
            self.assertAllEqual(outer_fn(x), 5.0 * (5.0 + 3.0))
            self.assertAllEqual(middle_fn(v, x), 3.0 * (3.0 + 5.0))
            (grad,) = gradients_impl.gradients(outer_fn(x), x)
            self.assertAllEqual(grad, 2 * 5.0 + 3.0)
            y = constant_op.constant(4.0)
            (grad,) = gradients_impl.gradients(outer_fn(y), y)
            self.assertAllEqual(grad, 2 * 4.0 + 3.0)
            self.evaluate(v.assign(constant_op.constant(1.5)))
            (grad,) = gradients_impl.gradients(outer_fn(y), y)
            self.assertAllEqual(grad, 2 * 4.0 + 1.5)
            (grad,) = gradients_impl.gradients(inner_fn(y, v), y)
            self.assertAllEqual(grad, 1.0)

    def testNestedDifferentiableFunctionNoneOutputs(self):
        if False:
            while True:
                i = 10

        @polymorphic_function.function
        def foo(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return (None, a * math_ops.add(a, b), None, 2 * a)

        @polymorphic_function.function
        def bar(x):
            if False:
                print('Hello World!')
            return foo(x, 1.0)
        x = constant_op.constant(5.0)
        with backprop.GradientTape(persistent=True) as tp:
            tp.watch(x)
            (none1, r1, none2, r2) = bar(x)
        g1 = tp.gradient(r1, x)
        g2 = tp.gradient(r2, x)
        self.assertAllEqual(r1, 30.0)
        self.assertAllEqual(r2, 10.0)
        self.assertIs(none1, None)
        self.assertIs(none2, None)
        self.assertAllEqual(g1, 2 * 5.0 + 1.0)
        self.assertAllEqual(g2, 2.0)

    def testGradientWithKeywordArguments(self):
        if False:
            return 10
        matmul = polymorphic_function.function(math_ops.matmul)

        def sq(x):
            if False:
                return 10
            return matmul(a=x, b=x, transpose_a=True)
        t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        (grad_t,) = backprop.gradients_function(sq, [0])(t)
        self.assertAllEqual(grad_t, [[6, 6], [14, 14]])
        with backprop.GradientTape(persistent=True) as tape:
            tape.watch(t)
            one = matmul(t, b=t, transpose_a=True)
            two = matmul(b=t, a=t, transpose_a=True)
            three = matmul(a=t, b=t, transpose_a=True)
        for output in [one, two, three]:
            self.assertAllEqual(tape.gradient(output, t), [[6, 6], [14, 14]])

    def testGradientInFunctionWithKeywordArguments(self):
        if False:
            print('Hello World!')

        @polymorphic_function.function
        def f(x):
            if False:
                i = 10
                return i + 15
            return backprop.gradients_function(lambda y: y * y, [0])(x)[0]
        self.assertAllEqual(f(x=constant_op.constant(1.0)), 2.0)

    def testFunctionHasNoSecondOrderGradient(self):
        if False:
            for i in range(10):
                print('nop')
        _ = nn_grad
        v = variables.Variable(1.0)

        @polymorphic_function.function
        def f(labels, logits):
            if False:
                return 10
            return polymorphic_function.function(nn_ops.sparse_softmax_cross_entropy_with_logits)(labels=labels, logits=logits + v)

        @polymorphic_function.function
        def f_grad():
            if False:
                for i in range(10):
                    print('nop')
            with backprop.GradientTape() as tape:
                logits = constant_op.constant([1.0, 2.0])
                tape.watch(logits)
                out = f(constant_op.constant(1), logits)
            return tape.gradient(out, logits)
        self.assertAllEqual([2], f_grad().shape)
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()