"""Tests for while_v2."""
from absl.testing import parameterized
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import importer
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.ops import while_v2
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.while_v2 import while_loop as while_loop_v2
from tensorflow.python.platform import test

def random_gamma(shape):
    if False:
        return 10
    return random_ops.random_gamma(shape, 1.0)

def random_gamma_with_alpha_beta(shape):
    if False:
        while True:
            i = 10
    return random_ops.random_gamma(shape, alpha=[[1.0], [3.0], [5.0], [6.0]], beta=[[3.0, 4.0]])

def random_poisson_v2(shape):
    if False:
        return 10
    return random_ops.random_poisson_v2(shape, 1.0)

def random_poisson_v2_with_lam(shape):
    if False:
        i = 10
        return i + 15
    return random_ops.random_poisson_v2(shape, [12.2, 3.3])

def fill(shape):
    if False:
        for i in range(10):
            print('nop')
    return array_ops.fill(shape, 1.0)

class WhileV2Test(test.TestCase, parameterized.TestCase):

    @test_util.run_deprecated_v1
    def testSingleLoopVar(self):
        if False:
            return 10
        x = constant_op.constant(2.0)
        ret = while_loop_v2(lambda v: v < 8.0, lambda v: v * v, [x], return_same_structure=False)
        grad = gradients_impl.gradients(ret, [x])
        with self.cached_session():
            self.assertEqual(self.evaluate(ret), 16.0)
            self.assertSequenceEqual(self.evaluate(grad), [32.0])

    @test_util.run_deprecated_v1
    def testSingleLoopVarBackPropFalse(self):
        if False:
            while True:
                i = 10
        x = constant_op.constant(2.0)
        ret = while_loop_v2(lambda v: v < 8.0, lambda v: v * v, [x], return_same_structure=False, back_prop=False)
        grad = gradients_impl.gradients(ret, [x])
        self.assertEqual(grad, [None])
        with self.cached_session():
            self.assertEqual(self.evaluate(ret), 16.0)

    @test_util.run_deprecated_v1
    def testCustomGradient(self):
        if False:
            return 10
        x = constant_op.constant(2.0)
        n = constant_op.constant(1.0, name='const-n')
        m = variables.Variable(1.0)
        self.evaluate(variables.global_variables_initializer())

        def body_fn(v):
            if False:
                while True:
                    i = 10

            @custom_gradient.custom_gradient
            def inner_fn(v):
                if False:
                    print('Hello World!')

                def grad_fn(dy, variables=None):
                    if False:
                        return 10
                    return (dy * 2 * v * n * m, [v * v])
                return (v * v * m, grad_fn)
            return inner_fn(v)
        ret = while_loop_v2(lambda v: v < 8.0, body_fn, [x], return_same_structure=False)
        grad = gradients_impl.gradients(ret, [x])
        with self.cached_session():
            self.assertEqual(self.evaluate(ret), 16.0)
            self.assertSequenceEqual(self.evaluate(grad), [32.0])

    @test_util.run_v1_only('b/120545219')
    def testReturnSameStructureTrue(self):
        if False:
            i = 10
            return i + 15
        x = constant_op.constant(2.0)
        ret = while_loop_v2(lambda v: v < 8.0, lambda v: v * v, [x], return_same_structure=True)
        grad = gradients_impl.gradients(ret, [x])
        with self.cached_session() as sess:
            eval_result = sess.run(ret)
            self.assertIsInstance(eval_result, list)
            self.assertLen(eval_result, 1)
            self.assertEqual(16.0, eval_result[0])
            self.assertSequenceEqual(sess.run(grad), [32.0])

    def testVerifyInputOutputTypesMatch(self):
        if False:
            return 10

        @def_function.function
        def BuildWhile():
            if False:
                while True:
                    i = 10
            x = constant_op.constant(1.0, dtypes.float32)

            def Body(x):
                if False:
                    print('Hello World!')
                return math_ops.cast(x, dtypes.float16) + 1
            while_loop_v2(lambda x: x < 10, Body, [x])
        with self.assertRaisesRegex(TypeError, "Loop var Const:0 enters the loop with type <dtype: 'float32'> but has type <dtype: 'float16'> after 1 iteration."):
            BuildWhile()

    @parameterized.parameters(dtypes.float32, dtypes.float64)
    def testGradientTapeResourceVariable(self, dtype):
        if False:
            i = 10
            return i + 15
        with context.eager_mode():
            v = variables.Variable(1.0, dtype=dtype)

            @def_function.function
            def fnWithLoop():
                if False:
                    while True:
                        i = 10
                with backprop.GradientTape() as tape:
                    (_, x) = while_loop_v2(lambda i, _: i < 2, lambda i, x: (i + 1, x * v), [0, constant_op.constant(2.0, dtype=dtype)])
                return tape.gradient(x, v)
            self.assertAllEqual(fnWithLoop(), 4.0)

    def testDeferredCaptures(self):
        if False:
            print('Hello World!')
        with context.eager_mode():
            c = constant_op.constant(10)

            @def_function.function
            def F():
                if False:
                    while True:
                        i = 10

                def Body(_):
                    if False:
                        for i in range(10):
                            print('nop')
                    return ops.get_default_graph().capture_call_time_value(lambda : c, tensor_spec.TensorSpec([], dtypes.int32))
                (x,) = while_loop_v2(lambda i: True, Body, [0], maximum_iterations=1)
                return x
            self.assertAllEqual(F(), 10)

    def checkIteratedGradients(self, func):
        if False:
            while True:
                i = 10
        with context.eager_mode():

            def _Grad(f):
                if False:
                    print('Hello World!')

                def _GradFunction(primal):
                    if False:
                        print('Hello World!')
                    with backprop.GradientTape() as tape:
                        tape.watch(primal)
                        primal_out = f(primal)
                    return tape.gradient(primal_out, primal)
                return _GradFunction
            f = func
            one = constant_op.constant(1.0)
            for _ in range(3):
                (theoretical, numerical) = gradient_checker_v2.compute_gradient(def_function.function(f), [one])
                self.assertAllClose(theoretical, numerical, rtol=0.001)
                f = _Grad(f)
                self.assertAllClose(array_ops.reshape(numerical, []), def_function.function(f)(one), rtol=0.001)

    def testIteratedGradients(self):
        if False:
            i = 10
            return i + 15

        def _Func(x):
            if False:
                while True:
                    i = 10
            (_, z) = while_loop_v2(lambda i, _: i < 2, lambda i, y: (i + 1, math_ops.cos(y)), [0, x])
            return z
        self.checkIteratedGradients(_Func)

    def testIteratedGradientsWithList(self):
        if False:
            for i in range(10):
                print('nop')

        def _Func(x):
            if False:
                print('Hello World!')
            results = list_ops.empty_tensor_list(element_shape=[], element_dtype=dtypes.float32)

            def _LoopBody(i, y, handle):
                if False:
                    print('Hello World!')
                return (i + 1, math_ops.cos(y), list_ops.tensor_list_push_back(handle, y))
            (_, z, results) = while_loop_v2(lambda i, _, h: i < 2, _LoopBody, [0, x, results])
            return z + math_ops.reduce_sum(list_ops.tensor_list_stack(results, dtypes.float32))
        self.checkIteratedGradients(_Func)

    def testGradWhileGradWhileWithVariable(self):
        if False:
            i = 10
            return i + 15
        with context.eager_mode():
            v = variables.Variable(1.0)

            @def_function.function
            def _Func(x):
                if False:
                    while True:
                        i = 10

                def _Inner(a):
                    if False:
                        i = 10
                        return i + 15
                    with backprop.GradientTape() as tape:
                        tape.watch(a)
                        (_, b) = while_loop_v2(lambda i, _: i < 2, lambda i, y: (i + 1, math_ops.cos(v + y)), [0, a])
                    return tape.gradient(b, a)
                (_, z) = while_loop_v2(lambda i, _: i < 2, lambda i, y: (i + 1, _Inner(y)), [0, x])
                return z
            with backprop.GradientTape(persistent=True) as tape:
                x = constant_op.constant(1.0)
                tape.watch(x)
                y = _Func(x)
            (dx, _) = tape.gradient(y, [x, v])
            (theoretical, numerical) = gradient_checker_v2.compute_gradient(_Func, [x])
            self.assertAllClose(numerical, theoretical, rtol=0.001)
            self.assertAllClose(array_ops.reshape(numerical, []), dx, rtol=0.001)

    def testThreeNestWithLists(self):
        if False:
            i = 10
            return i + 15
        with context.eager_mode():

            def _WrapInWhile(f):
                if False:
                    while True:
                        i = 10

                def _Wrapped(x):
                    if False:
                        i = 10
                        return i + 15
                    results = list_ops.empty_tensor_list(element_shape=[], element_dtype=dtypes.float32)

                    def _LoopBody(i, y, handle):
                        if False:
                            while True:
                                i = 10
                        return (i + 1, f(math_ops.cos(y)), list_ops.tensor_list_push_back(handle, y))
                    (_, z, results) = while_loop.while_loop(lambda i, _, h: i < 2, _LoopBody, [0, x, results])
                    return z + math_ops.reduce_sum(list_ops.tensor_list_stack(results, dtypes.float32))
                return _Wrapped
            f = math_ops.sin
            target_function = _WrapInWhile(_WrapInWhile(_WrapInWhile(f)))

            @def_function.function
            def _TapeFromGraphMode(x):
                if False:
                    print('Hello World!')
                with backprop.GradientTape(persistent=True) as tape:
                    tape.watch(x)
                    y = target_function(x)
                return tape.gradient(y, x)
            x = constant_op.constant(1.0)
            dx = _TapeFromGraphMode(x)
            (theoretical, numerical) = gradient_checker_v2.compute_gradient(target_function, [x])
            self.assertAllClose(numerical, theoretical, rtol=0.003)
            self.assertAllClose(array_ops.reshape(numerical, []), dx, rtol=0.003)

    def testDeviceLabelsInherited(self):
        if False:
            return 10

        def _LoopBody(i, y):
            if False:
                return 10
            result = math_ops.cos(y)
            self.assertIn('CPU:10', result.device)
            with ops.device('CPU:11'):
                result = array_ops.identity(result)
            self.assertIn('CPU:11', result.device)
            return (i + 1, result)

        @def_function.function
        def _FunctionWithWhileLoop():
            if False:
                return 10
            x = constant_op.constant(1.0)
            with ops.device('CPU:10'):
                (_, z) = while_loop_v2(lambda i, _: i < 2, _LoopBody, [0, x])
            return z
        _FunctionWithWhileLoop.get_concrete_function()

    def testExternalControlDependencies(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default(), self.test_session():
            v = variables.Variable(1.0)
            self.evaluate(v.initializer)
            op = v.assign_add(1.0)

            def body_fn(i):
                if False:
                    i = 10
                    return i + 15
                with ops.control_dependencies([op]):
                    return i + 1
            loop = while_loop_v2(lambda i: i < 1, body_fn, [0])
            loop[0].op.run()
            self.assertAllEqual(self.evaluate(v), 2.0)

    @test_util.run_deprecated_v1
    def testMultipleLoopVarsBasic(self):
        if False:
            print('Hello World!')
        x = constant_op.constant(5.0)
        y = constant_op.constant(3.0)
        ret = while_loop_v2(lambda v, _: v < 45.0, lambda v, w: (v * w, w), [x, y], return_same_structure=False)
        grad = gradients_impl.gradients(ret, [x])
        with self.cached_session():
            self.assertSequenceEqual(self.evaluate(ret), [45.0, 3.0])
            self.assertSequenceEqual(self.evaluate(grad), [9.0])

    @test_util.run_deprecated_v1
    def testMultipleLoopNonscalarCond(self):
        if False:
            while True:
                i = 10
        x = constant_op.constant([[5.0]])
        y = constant_op.constant(3.0)
        ret = while_loop_v2(lambda v, _: v < 45.0, lambda v, w: (v * w, w), [x, y], return_same_structure=False)
        grad = gradients_impl.gradients(ret, [x])
        with self.cached_session():
            self.assertSequenceEqual(self.evaluate(ret), [45.0, 3.0])
            self.assertSequenceEqual(self.evaluate(grad), [9.0])

    @test_util.run_deprecated_v1
    def testMultipleLoopVars(self):
        if False:
            for i in range(10):
                print('nop')
        x = constant_op.constant(5.0)
        y = constant_op.constant(3.0)
        ret = while_loop_v2(lambda v, _: v < 45.0, lambda v, w: (v * w, v + w), [x, y], return_same_structure=False)
        gradx_0 = gradients_impl.gradients(ret[0], [x])
        gradx_1 = gradients_impl.gradients(ret[1], [x])
        gradx_2 = gradients_impl.gradients(ret, [x])
        grady_0 = gradients_impl.gradients(ret[0], [y])
        grady_1 = gradients_impl.gradients(ret[1], [y])
        grady_2 = gradients_impl.gradients(ret, [y])
        with self.cached_session():
            self.assertSequenceEqual(self.evaluate(ret), [120.0, 23.0])
            self.assertSequenceEqual(self.evaluate(gradx_0), [39.0])
            self.assertSequenceEqual(self.evaluate(gradx_1), [4.0])
            self.assertSequenceEqual(self.evaluate(gradx_2), [43.0])
            self.assertSequenceEqual(self.evaluate(grady_0), [55.0])
            self.assertSequenceEqual(self.evaluate(grady_1), [6.0])
            self.assertSequenceEqual(self.evaluate(grady_2), [61.0])

    @test_util.run_deprecated_v1
    def testGradientTape(self):
        if False:
            while True:
                i = 10
        with backprop.GradientTape() as t:
            x = constant_op.constant(2.0)
            t.watch(x)
            ret = while_loop_v2(lambda v: v < 4.0, lambda v: v * v, [x], return_same_structure=False)
        grad = t.gradient(ret, x)
        with self.cached_session() as sess:
            self.assertAllEqual(sess.run(grad), 4.0)

    @test_util.run_deprecated_v1
    def testMultipleWhileLoops(self):
        if False:
            return 10
        x = constant_op.constant(2.0)
        ret1 = while_loop_v2(lambda v: v < 4.0, lambda v: v * v, [x], return_same_structure=False)
        ret2 = while_loop_v2(lambda v: v < 16.0, lambda v: v * v, [ret1], return_same_structure=False)
        grad = gradients_impl.gradients(ret2, [x])
        grad_grad = gradients_impl.gradients(grad, [x])
        with self.cached_session():
            self.assertSequenceEqual(self.evaluate(grad), [32.0])
            self.assertSequenceEqual(self.evaluate(grad_grad), [48.0])

    def testMultipleWhileLoopsWithFunc(self):
        if False:
            for i in range(10):
                print('nop')
        x = constant_op.constant(2.0)

        @def_function.function
        def Fn():
            if False:
                print('Hello World!')
            ret1 = while_loop_v2(lambda v: v < 4.0, lambda v: v * v, [x], return_same_structure=False, name='while_1')
            ret2 = while_loop_v2(lambda v: v < 16.0, lambda v: v * v, [x], return_same_structure=False, name='while_2')
            return (ret1, ret2)
        concrete_fn = Fn.get_concrete_function()
        while_1 = concrete_fn.graph.get_operation_by_name('while_1')
        while_2 = concrete_fn.graph.get_operation_by_name('while_2')
        self.assertEqual(while_1.type, 'StatelessWhile')
        self.assertEqual(while_2.type, 'StatelessWhile')
        self.assertEmpty(while_1.control_inputs)
        self.assertEmpty(while_2.control_inputs)

    def testMultipleWhileLoopsGradStateless(self):
        if False:
            i = 10
            return i + 15

        @def_function.function
        def Fn():
            if False:
                for i in range(10):
                    print('nop')
            x = constant_op.constant(2.0)
            with backprop.GradientTape() as tape:
                tape.watch(x)
                ret1 = while_loop_v2(lambda v: v < 4.0, lambda v: v * v, [x], return_same_structure=False, name='while_1')
                ret2 = while_loop_v2(lambda v: v < 16.0, lambda v: v * v, [x], return_same_structure=False, name='while_2')
                loss = ret1 + ret2
            return tape.gradient(loss, x)
        graph = Fn.get_concrete_function().graph
        while_ops = [op for op in graph.get_operations() if 'While' in op.type]
        self.assertAllEqual([op.type for op in while_ops], ['StatelessWhile'] * 4, 'Must have exactly 4 StatelessWhile ops.')
        for op in while_ops:
            self.assertEmpty(op.control_inputs, '{} should not have any control inputs'.format(op.name))

    def testMultipleWhileLoopsWithDeps(self):
        if False:
            return 10
        x = variables.Variable(2.0)
        c = constant_op.constant(2.0)

        @def_function.function
        def Fn():
            if False:
                i = 10
                return i + 15

            def Body1(v):
                if False:
                    i = 10
                    return i + 15
                x.assign(x)
                return v * x
            ret1 = while_loop_v2(lambda v: v < 4.0, Body1, [c], return_same_structure=False, name='while_1')

            def Body2(v):
                if False:
                    return 10
                x.assign(x)
                return v * x * x
            ret2 = while_loop_v2(lambda v: v < 16.0, Body2, [c], return_same_structure=False, name='while_2')
            return (ret1, ret2)
        concrete_fn = Fn.get_concrete_function()
        while_1 = concrete_fn.graph.get_operation_by_name('while_1')
        while_2 = concrete_fn.graph.get_operation_by_name('while_2')
        self.assertEqual(while_1.type, 'While')
        self.assertEqual(while_2.type, 'While')
        self.assertEmpty(while_1.control_inputs)
        self.assertLen(while_2.control_inputs, 1)
        self.assertIs(while_2.control_inputs[0], while_1)

    def testMultipleWhileLoopsWithVarsDeps(self):
        if False:
            return 10
        x1 = variables.Variable(2.0)
        x2 = variables.Variable(3.0)
        c = constant_op.constant(2.0)

        @def_function.function
        def Fn():
            if False:
                return 10

            def Body1(v):
                if False:
                    while True:
                        i = 10
                x1.assign(x1)
                return v * x1
            ret1 = while_loop_v2(lambda v: v < 4.0, Body1, [c], return_same_structure=False, name='while_1')

            def Body2(v):
                if False:
                    i = 10
                    return i + 15
                x1.assign(x1)
                return v * x1 * x1
            ret2 = while_loop_v2(lambda v: v < 16.0, Body2, [c], return_same_structure=False, name='while_2')

            def Body3(v):
                if False:
                    while True:
                        i = 10
                x2.assign(x2)
                return v * x2
            ret3 = while_loop_v2(lambda v: v < 4.0, Body3, [c], return_same_structure=False, name='while_3')

            def Body4(v):
                if False:
                    i = 10
                    return i + 15
                x2.assign(x2)
                return v * x2 * x2
            ret4 = while_loop_v2(lambda v: v < 16.0, Body4, [c], return_same_structure=False, name='while_4')
            ret5 = while_loop_v2(lambda v: v < 16.0, lambda v: v * v, [c], return_same_structure=False, name='while_stateless')
            return (ret1, ret2, ret3, ret4, ret5)
        concrete_fn = Fn.get_concrete_function()
        while_1 = concrete_fn.graph.get_operation_by_name('while_1')
        while_2 = concrete_fn.graph.get_operation_by_name('while_2')
        while_3 = concrete_fn.graph.get_operation_by_name('while_3')
        while_4 = concrete_fn.graph.get_operation_by_name('while_4')
        while_stateless = concrete_fn.graph.get_operation_by_name('while_stateless')
        self.assertEqual(while_1.type, 'While')
        self.assertEqual(while_2.type, 'While')
        self.assertEqual(while_3.type, 'While')
        self.assertEqual(while_4.type, 'While')
        self.assertEqual(while_stateless.type, 'StatelessWhile')
        self.assertEmpty(while_1.control_inputs)
        self.assertLen(while_2.control_inputs, 1)
        self.assertIs(while_2.control_inputs[0], while_1)
        self.assertEmpty(while_3.control_inputs)
        self.assertLen(while_4.control_inputs, 1)
        self.assertIs(while_4.control_inputs[0], while_3)
        self.assertEmpty(while_stateless.control_inputs)

    @test_util.run_deprecated_v1
    def testDoubleDerivative(self):
        if False:
            return 10
        x = constant_op.constant(2.0)
        ret = while_loop_v2(lambda v: v < 8.0, lambda v: v ** 2, [x], return_same_structure=False)
        grad = gradients_impl.gradients(ret, [x])
        grad_grad = gradients_impl.gradients(grad, [x])
        with self.cached_session():
            self.assertEqual(self.evaluate(ret), 16.0)
            self.assertSequenceEqual(self.evaluate(grad), [32.0])
            self.assertSequenceEqual(self.evaluate(grad_grad), [48.0])

    @test_util.run_v2_only
    def testMultipleWhileLoopsEager(self):
        if False:
            return 10

        @def_function.function
        def Func():
            if False:
                print('Hello World!')
            x = constant_op.constant(2.0)
            ret1 = while_loop_v2(lambda v: v < 4.0, lambda v: v * v, [x], return_same_structure=False)
            ret2 = while_loop_v2(lambda v: v < 16.0, lambda v: v * v, [ret1], return_same_structure=False)
            grad = gradients_impl.gradients(ret2, [x])[0]
            grad_grad = gradients_impl.gradients(grad, [x])[0]
            return (grad, grad_grad)
        (grad, grad_grad) = Func()
        self.assertEqual(grad.numpy(), 32.0)
        self.assertEqual(grad_grad.numpy(), 48.0)

    @test_util.run_v2_only
    def testDoubleDerivativeEager(self):
        if False:
            return 10

        @def_function.function
        def Func():
            if False:
                while True:
                    i = 10
            x = constant_op.constant(2.0)
            ret = while_loop_v2(lambda v: v < 8.0, lambda v: v ** 2, [x], return_same_structure=False)
            grad = gradients_impl.gradients(ret, [x])[0]
            grad_grad = gradients_impl.gradients(grad, [x])[0]
            return (ret, grad, grad_grad)
        (ret, grad, grad_grad) = Func()
        self.assertEqual(ret.numpy(), 16.0)
        self.assertEqual(grad.numpy(), 32.0)
        self.assertEqual(grad_grad.numpy(), 48.0)

    def _testPruning(self):
        if False:
            print('Hello World!')
        x = constant_op.constant(1)
        tensor_list = list_ops.empty_tensor_list(element_dtype=x.dtype, element_shape=x.shape)

        def Cond(x, tl):
            if False:
                for i in range(10):
                    print('nop')
            del tl
            return x < 5

        def Body(x, tl):
            if False:
                i = 10
                return i + 15
            return (x + 1, list_ops.tensor_list_push_back(tl, x))
        outputs = while_loop.while_loop(Cond, Body, [x, tensor_list])
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(outputs[0])
        g = GetOptimizedGraph()
        enter_count = 2 if control_flow_util.ENABLE_CONTROL_FLOW_V2 else 1
        self.assertLen([n for n in g.node if n.op == 'Enter'], enter_count)
        self.assertEmpty([n for n in g.node if n.op == 'Enter' and n.attr['T'].type == dtypes.variant.as_datatype_enum])
        self.assertEmpty([n for n in g.node if n.op == 'TensorListPushBack'])
        stack = list_ops.tensor_list_stack(outputs[1], element_dtype=x.dtype)
        train_op.append(stack)
        g = GetOptimizedGraph()
        enter_count = 3 if control_flow_util.ENABLE_CONTROL_FLOW_V2 else 2
        self.assertLen([n for n in g.node if n.op == 'Enter'], enter_count)
        self.assertNotEmpty([n for n in g.node if n.op == 'Enter' and n.attr['T'].type == dtypes.variant.as_datatype_enum])
        self.assertNotEmpty([n for n in g.node if n.op == 'TensorListPushBack'])

    @test_util.run_deprecated_v1
    def testPruningV1(self):
        if False:
            i = 10
            return i + 15
        self._testPruning()

    @test_util.enable_control_flow_v2
    @test_util.run_deprecated_v1
    def testPruningV2(self):
        if False:
            i = 10
            return i + 15
        self._testPruning()

    def _testDoNotAccumulateInvariants(self):
        if False:
            return 10
        push_op = 'TensorListPushBack' if control_flow_v2_toggles.control_flow_v2_enabled() else 'StackPushV2'
        v = constant_op.constant(5.0, name='v')
        r = while_loop.while_loop(lambda _: True, lambda x: v * x, [1.0], maximum_iterations=5)
        output = gradients_impl.gradients(r, v)[0]
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(output)
        g = GetOptimizedGraph()
        self.assertLen([n for n in g.node if n.op == push_op], 1)

    @test_util.run_deprecated_v1
    def testDoNotAccumulateInvariantsV1(self):
        if False:
            for i in range(10):
                print('nop')
        self._testDoNotAccumulateInvariants()

    @test_util.run_deprecated_v1
    @test_util.enable_control_flow_v2
    def testDoNotAccumulateInvariantsV2(self):
        if False:
            return 10
        self._testDoNotAccumulateInvariants()

    @test_util.enable_control_flow_v2
    @test_util.run_deprecated_v1
    @test_util.enable_output_all_intermediates
    def testPruningNested(self):
        if False:
            while True:
                i = 10
        assert control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
        x = constant_op.constant(0)
        tensor_list = list_ops.empty_tensor_list(element_dtype=x.dtype, element_shape=x.shape)

        def Cond(x, tl):
            if False:
                print('Hello World!')
            del tl
            return x < 25

        def Body(x, tl):
            if False:
                return 10

            def InnerCond(inner_x, unused_outer_x, unused_tl):
                if False:
                    while True:
                        i = 10
                return inner_x < 5

            def InnerBody(inner_x, outer_x, tl):
                if False:
                    for i in range(10):
                        print('nop')
                return (inner_x + 1, outer_x + 1, list_ops.tensor_list_push_back(tl, x))
            inner_x = constant_op.constant(0)
            return while_loop.while_loop(InnerCond, InnerBody, [inner_x, x, tl])[1:]
        outputs = while_loop.while_loop(Cond, Body, [x, tensor_list])
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(outputs[0])
        g = GetOptimizedGraph()
        self.assertEmpty([n for n in g.node if n.op == 'Enter' and n.attr['T'].type == dtypes.variant.as_datatype_enum])
        self.assertEmpty([n for n in g.node if n.op == 'TensorListPushBack'])
        self.assertEmpty([n for n in g.node if n.op == '_While'])
        stack = list_ops.tensor_list_stack(outputs[1], element_dtype=x.dtype)
        train_op.append(stack)
        g = GetOptimizedGraph()
        self.assertNotEmpty([n for n in g.node if n.op == 'Enter' and n.attr['T'].type == dtypes.variant.as_datatype_enum])
        self.assertNotEmpty([n for n in g.node if n.op == 'TensorListPushBack'])

    @test_util.enable_control_flow_v2
    @test_util.run_deprecated_v1
    @test_util.enable_output_all_intermediates
    def testPruningNested2(self):
        if False:
            print('Hello World!')
        assert control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
        v = constant_op.constant(5.0, name='v')
        p = array_ops.placeholder(dtype=dtypes.int32)

        def MidBodyBuilder(iterations):
            if False:
                while True:
                    i = 10

            def MidBody(i, x):
                if False:
                    for i in range(10):
                        print('nop')
                r = while_loop.while_loop(lambda *_: True, lambda i, x: (i + 1, math_ops.multiply(v, x, name='my_mul')), (0, x), maximum_iterations=iterations, name='inner')
                return (i + 1, gradients_impl.gradients(x + r[1], v)[0])
            return MidBody

        def OuterBody(i, x):
            if False:
                for i in range(10):
                    print('nop')
            iterations = array_ops.size(p, name='iterations')
            return (i + 1, x + while_loop.while_loop(lambda *_: True, MidBodyBuilder(iterations), (0, x), maximum_iterations=iterations, name='mid')[1])

        def CreateWhileLoop():
            if False:
                print('Hello World!')
            with ops.device('/cpu:0'):
                r = while_loop.while_loop(lambda *_: True, OuterBody, (0, 1.0), maximum_iterations=5, name='outer')
                return array_ops.identity(r[1])
        output = CreateWhileLoop()
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(output)
        g = GetOptimizedGraph()
        self.assertLen([n for n in g.node if n.op == 'TensorListPushBack'], 1)

    @test_util.enable_control_flow_v2
    @test_util.run_deprecated_v1
    @test_util.enable_output_all_intermediates
    def testPruningNested3(self):
        if False:
            for i in range(10):
                print('nop')
        assert control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
        v = constant_op.constant(5.0, name='v')

        def CreateWhileLoop():
            if False:
                return 10
            r = while_loop.while_loop(lambda _: True, lambda x: math_ops.multiply(v, x, name='my_mul'), [1.0], maximum_iterations=5, name='outer')
            return array_ops.identity(r)
        r = CreateWhileLoop()
        output = gradients_impl.gradients(r, v)[0]
        train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
        train_op.append(output)
        g = GetOptimizedGraph()
        self.assertLen([n for n in g.node if n.op == 'TensorListPushBack'], 1)

    def _assertNotAccumulated(self, while_op, index):
        if False:
            for i in range(10):
                print('nop')
        'Asserts that `while_op` input at `index` is not accumulated.'
        body_graph = while_v2._get_graph(while_op, 'body', '_body_graph')
        placeholder = body_graph.inputs[index]
        self.assertNotIn('TensorListPushBack', [op.type for op in placeholder.consumers()])

    @test_util.enable_control_flow_v2
    @test_util.run_deprecated_v1
    @test_util.enable_output_all_intermediates
    def testDoNotOutputLoopCounterAsIntermediate(self):
        if False:
            for i in range(10):
                print('nop')
        assert control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
        v = constant_op.constant(5.0, name='v')
        r = while_loop.while_loop(lambda _: True, lambda x: v * x, [1.0], maximum_iterations=5)
        while_op = r.op.inputs[0].op
        self._assertNotAccumulated(while_op, 0)

    @test_util.enable_control_flow_v2
    @test_util.run_deprecated_v1
    @test_util.enable_output_all_intermediates
    def testDoNotOutputLoopInvariantAsIntermediate(self):
        if False:
            i = 10
            return i + 15
        assert control_flow_util_v2._EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE

        def GetInputIndex(op, tensor):
            if False:
                while True:
                    i = 10
            for (index, inp) in enumerate(op.inputs):
                if inp is tensor:
                    return index
        v = constant_op.constant(5.0, name='v')
        r = while_loop.while_loop(lambda _: True, lambda x: v * x, [1.0], maximum_iterations=5)
        while_op = r.op.inputs[0].op
        index = GetInputIndex(while_op, v)
        self._assertNotAccumulated(while_op, index)

    @test_util.run_deprecated_v1
    def testCaptureExternalTensorInCond(self):
        if False:
            i = 10
            return i + 15
        x = constant_op.constant(2.0)
        y = constant_op.constant(1.0)
        ret = while_loop_v2(lambda v: v + y < 9.0, lambda v: v * 3.0, [x], return_same_structure=False)
        grad = gradients_impl.gradients(ret, [x])
        with self.cached_session():
            self.assertEqual(self.evaluate(ret), 18.0)
            self.assertSequenceEqual(self.evaluate(grad), [9.0])

    @test_util.run_deprecated_v1
    def testCaptureExternalTensorInBody(self):
        if False:
            return 10
        x = constant_op.constant(2.0)
        y = constant_op.constant(3.0)
        ret = while_loop_v2(lambda v: v < 8.0, lambda v: v * y, [x], return_same_structure=False)
        grad = gradients_impl.gradients(ret, [x])
        with self.cached_session():
            self.assertEqual(self.evaluate(ret), 18.0)
            self.assertSequenceEqual(self.evaluate(grad), [9.0])

    @test_util.run_deprecated_v1
    def testLoopWithTensorListPushBack(self):
        if False:
            i = 10
            return i + 15
        x = constant_op.constant(2.0)
        tensor_list = list_ops.empty_tensor_list(element_dtype=dtypes.float32, element_shape=ScalarShape())

        def Cond(x, tl):
            if False:
                print('Hello World!')
            del tl
            return x < 5.0

        def Body(x, tl):
            if False:
                return 10
            tl = list_ops.tensor_list_push_back(tl, x)
            tl = list_ops.tensor_list_push_back(tl, constant_op.constant(100.0))
            return (x ** 2.0, tl)
        ret = while_loop_v2(Cond, Body, [x, tensor_list], return_same_structure=False)
        grad = gradients_impl.gradients(ret[0], x)
        with self.cached_session() as sess:
            self.assertEqual(sess.run(ret[0]), 16.0)
            self.assertSequenceEqual(self.evaluate(grad), [32.0])

    @test_util.run_deprecated_v1
    def testDuplicateAccumulator(self):
        if False:
            return 10
        x = constant_op.constant(2.0)
        tensor_list = list_ops.empty_tensor_list(element_dtype=dtypes.float32, element_shape=ScalarShape())

        def Cond(x, tl):
            if False:
                return 10
            del tl
            return x < 5.0

        def Body(x, tl):
            if False:
                i = 10
                return i + 15
            tl = list_ops.tensor_list_push_back(tl, x)
            return (x ** 2.0, tl)
        ret = while_loop_v2(Cond, Body, [x, tensor_list], return_same_structure=False)
        for op in ops.get_default_graph().get_operations():
            if op.type == 'While' or op.type == 'StatelessWhile':
                while_op = op
        body_graph = while_v2._get_graph(while_op, 'body', '_body_graph')
        x_input_index = [i for (i, inp) in enumerate(while_op.inputs) if inp == x][0]
        x_input_t = body_graph.inputs[x_input_index]
        accumulator_count = len([c for c in x_input_t.consumers() if c.type == 'TensorListPushBack'])
        self.assertEqual(accumulator_count, 1)
        grad = gradients_impl.gradients(ret[0], x)
        with self.cached_session() as sess:
            self.assertEqual(sess.run(ret[0]), 16.0)
            self.assertSequenceEqual(self.evaluate(grad), [32.0])

    @parameterized.named_parameters(('UnknownShape', None), ('PartiallyDefinedShape', [None, 2]), ('FullyDefinedShape', [1, 2]))
    @test_util.run_deprecated_v1
    def testAccumulatorElementShape(self, shape):
        if False:
            print('Hello World!')

        def MatchShape(actual_tensor_shape):
            if False:
                i = 10
                return i + 15
            if shape is None:
                self.assertIsNone(actual_tensor_shape.dims)
            else:
                self.assertListEqual(actual_tensor_shape.as_list(), shape)

        def GetAccumulatorForInputAtIndex(while_op, idx):
            if False:
                return 10
            body_graph = while_v2._get_graph(while_op, 'body', '_body_graph')
            y_input_t = body_graph.inputs[idx]
            push_back_node = [c for c in y_input_t.consumers() if c.type == 'TensorListPushBack'][0]
            output_idx = body_graph.outputs.index(push_back_node.outputs[0])
            return while_op.outputs[output_idx]
        x = array_ops.placeholder(dtype=dtypes.float32, shape=shape)
        y = array_ops.placeholder(dtype=dtypes.float32, shape=shape)
        ret = while_loop_v2(lambda v, u: v < 8.0, lambda v, u: (math_ops.pow(v, u), u), [x, y], return_same_structure=True)
        while_op = ret[0].op.inputs[0].op
        grad = gradients_impl.gradients(ret[0], x)
        grad_while_op = grad[0].op.inputs[0].op
        x_input_index = [i for (i, inp) in enumerate(while_op.inputs) if x == inp][0]
        output = GetAccumulatorForInputAtIndex(while_op, x_input_index)
        (_, val) = list_ops.tensor_list_pop_back(output, element_dtype=dtypes.float32)
        MatchShape(val.shape)
        gradients_impl.gradients(grad, x)
        grad_output_index = grad_while_op.outputs.index(grad[0].op.inputs[0])
        grad_output = GetAccumulatorForInputAtIndex(grad_while_op, grad_output_index)
        (_, val) = list_ops.tensor_list_pop_back(grad_output, element_dtype=dtypes.float32)
        MatchShape(val.shape)

    def _createWhile(self, name):
        if False:
            i = 10
            return i + 15
        'Helper function testDefaultName.'
        output = while_v2.while_loop(lambda i: i < 3, lambda i: i + 1, [constant_op.constant(0)], return_same_structure=False)
        while_op = output.op.inputs[0].op
        self.assertEqual(while_op.type, 'StatelessWhile')
        return while_op

    def testDefaultName(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default():
            while_op = self._createWhile(None)
            self.assertEqual(while_op.name, 'while')
            self.assertRegex(while_op.get_attr('cond').name, 'while_cond_\\d*')
            self.assertRegex(while_op.get_attr('body').name, 'while_body_\\d*')
        with ops.Graph().as_default():
            with ops.name_scope('foo'):
                while1_op = self._createWhile('')
                self.assertEqual(while1_op.name, 'foo/while')
                self.assertRegex(while1_op.get_attr('cond').name, 'foo_while_cond_\\d*')
                self.assertRegex(while1_op.get_attr('body').name, 'foo_while_body_\\d*')
                while2_op = self._createWhile(None)
                self.assertEqual(while2_op.name, 'foo/while_1')
                self.assertRegex(while2_op.get_attr('cond').name, 'foo_while_1_cond_\\d*')
                self.assertRegex(while2_op.get_attr('body').name, 'foo_while_1_body_\\d*')

    @test_util.enable_control_flow_v2
    @test_util.run_deprecated_v1
    def testWhileAndTensorArray(self):
        if False:
            while True:
                i = 10
        param = constant_op.constant(2.0)
        y0 = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='elems')
        r = map_fn.map_fn(lambda x: math_ops.multiply(x, param), y0)
        grad = gradients_impl.gradients(r, param)[0]
        self.assertAllClose([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], self.evaluate(r))
        self.assertAllClose(21.0, self.evaluate(grad))

    @test_util.run_deprecated_v1
    def testNestedWhile(self):
        if False:
            i = 10
            return i + 15
        n = constant_op.constant(3.0)
        m = constant_op.constant(5.0)
        sum_of_powers = constant_op.constant(0.0)

        def Body(i, previous_sum):
            if False:
                i = 10
                return i + 15
            prod = constant_op.constant(1.0)
            return (i - 1.0, previous_sum + while_loop_v2(lambda c, _: c > 0, lambda c, v: (c - 1.0, v * n), [i, prod], return_same_structure=False)[1])
        result = while_loop_v2(lambda i, _: i >= 0, Body, [m, sum_of_powers], return_same_structure=False)[1]
        grad = gradients_impl.gradients(result, [n])
        self.assertEqual(self.evaluate(result), 364.0)
        self.assertSequenceEqual(self.evaluate(grad), [547.0])

    @test_util.run_deprecated_v1
    def testNestedWhileWithLegacyDefun(self):
        if False:
            for i in range(10):
                print('nop')
        n = constant_op.constant(3.0)
        m = constant_op.constant(5.0)
        sum_of_powers = constant_op.constant(0.0)

        def Body(i, previous_sum):
            if False:
                while True:
                    i = 10
            prod = constant_op.constant(1.0)

            def InnerBodyWrapper(c, v):
                if False:
                    for i in range(10):
                        print('nop')

                @function.Defun(dtypes.float32, dtypes.float32)
                def InnerBody(c, v):
                    if False:
                        i = 10
                        return i + 15
                    return (c - 1.0, v * n)
                results = InnerBody(c, v)
                results[0].set_shape([])
                results[1].set_shape([])
                return results
            return (i - 1.0, previous_sum + while_loop_v2(lambda c, _: c > 0, InnerBodyWrapper, [i, prod], return_same_structure=False)[1])
        result = while_loop_v2(lambda i, _: i >= 0, Body, [m, sum_of_powers], return_same_structure=False)[1]
        grad = gradients_impl.gradients(result, [n])
        self.assertEqual(self.evaluate(result), 364.0)
        self.assertSequenceEqual(self.evaluate(grad), [547.0])

    @test_util.run_deprecated_v1
    def testIdentityNodeInBody(self):
        if False:
            i = 10
            return i + 15

        def Body(v):
            if False:
                while True:
                    i = 10
            v = array_ops.identity(v)
            v = array_ops.identity(v)
            return v * v
        x = constant_op.constant(2.0)
        ret = while_loop_v2(lambda v: v < 8.0, Body, [x], return_same_structure=False)
        grad = gradients_impl.gradients(ret, [x])
        self.assertEqual(self.evaluate(ret), 16.0)
        self.assertSequenceEqual(self.evaluate(grad), [32.0])

    @test_util.run_deprecated_v1
    def testForwardPassRewrite(self):
        if False:
            i = 10
            return i + 15
        x = constant_op.constant(1.0, name='x')
        output = while_v2.while_loop(lambda x: x < 10.0, lambda x: x * 2.0, [x])[0]
        while_op = output.op.inputs[0].op
        self.assertEqual(while_op.type, 'StatelessWhile')
        self.assertLen(while_op.outputs, 3)
        gradients_impl.gradients(output, x)
        self.assertLen(while_op.outputs, 4)
        gradients_impl.gradients(output, x)
        self.assertLen(while_op.outputs, 4)

    @parameterized.named_parameters(('RandomUniform', random_ops.random_uniform, [5, 3]), ('RandomNormal', random_ops.random_normal, [5, 3]), ('ParameterizedTruncatedNormal', random_ops.parameterized_truncated_normal, [5, 3]), ('TruncatedNormal', random_ops.truncated_normal, [5, 3]), ('RandomGamma', random_gamma, [5, 3]), ('RandomPoissonV2', random_poisson_v2, [5, 3]), ('RandomGammaWithAlphaBeta', random_gamma_with_alpha_beta, [5, 3, 4, 2]), ('RandomPoissonV2WithLam', random_poisson_v2_with_lam, [5, 3, 2]))
    @test_util.run_deprecated_v1
    def testRandomOpsShape(self, random_fn, expected_shape):
        if False:
            print('Hello World!')
        shape = constant_op.constant([3])

        def Body(i, u):
            if False:
                print('Hello World!')
            shape_extended = array_ops.concat([[5], shape], axis=0)
            u = random_fn(shape_extended)
            assert u.shape.as_list() == expected_shape, str(u.shape.as_list())
            return (i + 1, u)
        (_, _) = while_loop_v2(cond=lambda i, _: i < 3, body=Body, loop_vars=[0, array_ops.zeros(expected_shape, dtype=dtypes.float32)])

    @test_util.run_deprecated_v1
    def testReshapeShape(self):
        if False:
            for i in range(10):
                print('nop')
        shape = constant_op.constant([3, 4])

        def Body(i, u):
            if False:
                print('Hello World!')
            shape_extended = array_ops.concat([[5], shape], axis=0)
            u = array_ops.reshape(u, [-1])
            assert u.shape.as_list() == [60], str(u.shape.as_list())
            u = array_ops.reshape(u, shape_extended)
            assert u.shape.as_list() == [5, 3, 4], str(u.shape.as_list())
            return (i + 1, u)
        (_, _) = while_loop_v2(cond=lambda i, _: i < 3, body=Body, loop_vars=[0, array_ops.zeros([5, 3, 4], dtype=dtypes.float32)])

    @parameterized.named_parameters(('Zeros', array_ops.zeros), ('Ones', array_ops.ones), ('Fill', fill))
    @test_util.run_deprecated_v1
    def testFillOpsShape(self, fill_fn):
        if False:
            while True:
                i = 10
        shape = constant_op.constant([3, 4])

        def Body(i, u):
            if False:
                for i in range(10):
                    print('nop')
            shape_extended = array_ops.concat([[5], shape], axis=0)
            u = fill_fn(shape_extended)
            assert u.shape.as_list() == [5, 3, 4], str(u.shape.as_list())
            return (i + 1, u)
        (_, _) = while_loop_v2(cond=lambda i, _: i < 3, body=Body, loop_vars=[0, array_ops.zeros([5, 3, 4], dtype=dtypes.float32)])

    @test_util.run_deprecated_v1
    def testExternalColocationGrad(self):
        if False:
            print('Hello World!')
        external_t = constant_op.constant(2.0)
        v0 = constant_op.constant(2.0)

        def Body(v):
            if False:
                return 10
            with ops.colocate_with(external_t):
                return v * v
        ret = while_loop_v2(lambda v: v < 8.0, Body, [v0])[0]
        grad = gradients_impl.gradients(ret, [v0])[0]
        self.assertAllEqual(ret, 16.0)
        self.assertAllEqual(grad, 32.0)

    @test_util.run_deprecated_v1
    def testDoNotAccumulateConstNodes(self):
        if False:
            i = 10
            return i + 15

        def Body(v):
            if False:
                print('Hello World!')
            return v * 2.0
        v0 = constant_op.constant(2.0)
        ret = while_loop_v2(lambda v: v < 8.0, Body, [v0])[0]
        unused_grad = gradients_impl.gradients(ret, [v0])[0]
        forward_while_op = ret.op.inputs[0].op
        body_graph = while_v2._get_graph(forward_while_op, 'body', '_body_graph')
        push_back_nodes = [o for o in body_graph.get_operations() if o.type == 'TensorListPushBack']
        self.assertLen(push_back_nodes, 1)

    def testDoNotAccumulateForwardTensorsForReductionOps(self):
        if False:
            print('Hello World!')

        @def_function.function
        def Fn():
            if False:
                return 10
            with backprop.GradientTape() as tape:
                x = constant_op.constant(2.0)
                tape.watch(x)

                def Body(i, x):
                    if False:
                        for i in range(10):
                            print('nop')
                    forward_graph = ops.get_default_graph()

                    @custom_gradient.custom_gradient
                    def SquaredWithZeroGrad(x):
                        if False:
                            for i in range(10):
                                print('nop')

                        def Grad(unused_g, variables=None):
                            if False:
                                print('Hello World!')
                            del variables
                            gradient_graph = ops.get_default_graph()
                            shape = gen_array_ops.shape(x)
                            assert shape.graph is forward_graph
                            rank = gen_array_ops.rank(x)
                            assert rank.graph is forward_graph
                            size = gen_array_ops.size(x)
                            assert size.graph is forward_graph
                            zeros = array_ops.zeros(shape)
                            assert zeros.graph is gradient_graph
                            return zeros
                        return (x * 2, Grad)
                    return (i + 1, SquaredWithZeroGrad(x))
                (_, result) = while_loop_v2(lambda i, _: i < 2, Body, [0, x])
            grad = tape.gradient(result, x)
            return grad
        Fn()

    def testDoNotAccumulateForwardTensorsForTensorListReductionOps(self):
        if False:
            return 10

        @def_function.function
        def Fn():
            if False:
                while True:
                    i = 10
            with backprop.GradientTape() as tape:
                e = constant_op.constant(2.0)
                x = list_ops.empty_tensor_list(element_dtype=dtypes.float32, element_shape=e.shape)
                x = list_ops.tensor_list_push_back(x, e)
                tape.watch(x)

                def Body(i, x):
                    if False:
                        while True:
                            i = 10
                    forward_graph = ops.get_default_graph()

                    @custom_gradient.custom_gradient
                    def IdentityWithZeroGrad(x):
                        if False:
                            while True:
                                i = 10

                        def Grad(unused_g, variables=None):
                            if False:
                                i = 10
                                return i + 15
                            del variables
                            gradient_graph = ops.get_default_graph()
                            shape = gen_list_ops.tensor_list_element_shape(x, shape_type=dtypes.int32)
                            assert shape.graph is forward_graph
                            size = gen_list_ops.tensor_list_length(x)
                            assert size.graph is forward_graph
                            zeros = gen_list_ops.tensor_list_reserve(shape, size, dtypes.float32)
                            assert zeros.graph is gradient_graph
                            return zeros
                        return (x, Grad)
                    return (i + 1, IdentityWithZeroGrad(x))
                (_, result) = while_loop_v2(lambda i, _: i < 2, Body, [0, x])
            ones_like = list_ops.tensor_list_from_tensor(array_ops.ones_like(list_ops.tensor_list_stack(result, element_dtype=dtypes.float32)), element_shape=tensor_shape.TensorShape([]))
            grad = tape.gradient(result, x, output_gradients=[ones_like])
            return grad
        Fn()

    @test_util.run_v2_only
    def testInheritParentNameScope(self):
        if False:
            return 10

        @def_function.function
        def F():
            if False:
                i = 10
                return i + 15
            with ops.name_scope('foo'):

                def Cond(unused_i):
                    if False:
                        i = 10
                        return i + 15
                    with ops.name_scope('cond'):
                        actual_name_scope = ops.get_name_scope()
                        expected_name_scope = 'foo/while/cond'
                        assert actual_name_scope == expected_name_scope, '%s does not match %s' % (actual_name_scope, expected_name_scope)
                    return False

                def Body(i):
                    if False:
                        print('Hello World!')
                    with ops.name_scope('body'):
                        actual_name_scope = ops.get_name_scope()
                        expected_name_scope = 'foo/while/body'
                        assert actual_name_scope == expected_name_scope, '%s does not match %s' % (actual_name_scope, expected_name_scope)
                    return i
                return while_v2.while_loop(Cond, Body, [0.0])
        F()

    @test_util.run_deprecated_v1
    def testDisableLowering(self):
        if False:
            print('Hello World!')
        old = control_flow_util_v2._DISABLE_LOWER_USING_SWITCH_MERGE
        control_flow_util_v2._DISABLE_LOWER_USING_SWITCH_MERGE = True
        with self.session() as sess:
            x = constant_op.constant(2.0)
            ret = while_loop_v2(lambda v: v < 8.0, lambda v: v * v, [x], return_same_structure=False)
            opts = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
            run_metadata = config_pb2.RunMetadata()
            self.assertEqual(sess.run(ret, options=opts, run_metadata=run_metadata), 16)
            for dev_stat in run_metadata.step_stats.dev_stats:
                for ns in dev_stat.node_stats:
                    self.assertNotIn('switch', ns.node_name)
        control_flow_util_v2._DISABLE_LOWER_USING_SWITCH_MERGE = old

    def _runBasicWithConfig(self, config):
        if False:
            for i in range(10):
                print('nop')
        with ops.device('/cpu:0'):
            x = constant_op.constant(0)
            (ret,) = while_loop_v2(lambda x: x < 1000, lambda x: x + 1, [x])
        with self.cached_session(config=config):
            self.assertEqual(1000, self.evaluate(ret))

    @test_util.run_deprecated_v1
    def testRunKernelsInline(self):
        if False:
            i = 10
            return i + 15
        config = config_pb2.ConfigProto()
        config.inter_op_parallelism_threads = -1
        self._runBasicWithConfig(config)

    @test_util.run_deprecated_v1
    def testSingleThreadedExecution(self):
        if False:
            return 10
        config = config_pb2.ConfigProto()
        config.experimental.executor_type = 'SINGLE_THREADED_EXECUTOR'
        self._runBasicWithConfig(config)

    def testIsControlFlowGraph(self):
        if False:
            i = 10
            return i + 15
        x = constant_op.constant(0)

        @def_function.function
        def F(c):
            if False:
                return 10

            def Cond(i):
                if False:
                    for i in range(10):
                        print('nop')
                self.assertTrue(i.graph.is_control_flow_graph)
                return i < 2

            def Body(i):
                if False:
                    i = 10
                    return i + 15
                i = i + 1
                self.assertTrue(i.graph.is_control_flow_graph)
                return i
            return while_loop_v2(Cond, Body, [c])
        (ret,) = F(x)
        self.assertEqual(2, self.evaluate(ret))

    def testImportFromSerializedWithFunctionInBody(self):
        if False:
            while True:
                i = 10
        serialized = 'node {\n      name: "Const"\n      op: "Const"\n      attr {\n        key: "dtype"\n        value {\n          type: DT_FLOAT\n        }\n      }\n      attr {\n        key: "value"\n        value {\n          tensor {\n            dtype: DT_FLOAT\n            tensor_shape {\n            }\n            float_val: 1.0\n          }\n        }\n      }\n    }\n    node {\n      name: "while/maximum_iterations"\n      op: "Const"\n      attr {\n        key: "dtype"\n        value {\n          type: DT_INT32\n        }\n      }\n      attr {\n        key: "value"\n        value {\n          tensor {\n            dtype: DT_INT32\n            tensor_shape {\n            }\n            int_val: -1\n          }\n        }\n      }\n    }\n    node {\n      name: "while/loop_counter"\n      op: "Const"\n      attr {\n        key: "dtype"\n        value {\n          type: DT_INT32\n        }\n      }\n      attr {\n        key: "value"\n        value {\n          tensor {\n            dtype: DT_INT32\n            tensor_shape {\n            }\n            int_val: 0\n          }\n        }\n      }\n    }\n    node {\n      name: "while"\n      op: "StatelessWhile"\n      input: "while/loop_counter"\n      input: "while/maximum_iterations"\n      input: "Const"\n      attr {\n        key: "T"\n        value {\n          list {\n            type: DT_INT32\n            type: DT_INT32\n            type: DT_FLOAT\n          }\n        }\n      }\n      attr {\n        key: "_lower_using_switch_merge"\n        value {\n          b: true\n        }\n      }\n      attr {\n        key: "_num_original_outputs"\n        value {\n          i: 3\n        }\n      }\n      attr {\n        key: "_read_only_resource_inputs"\n        value {\n          list {\n          }\n        }\n      }\n      attr {\n        key: "body"\n        value {\n          func {\n            name: "while_body_822"\n          }\n        }\n      }\n      attr {\n        key: "cond"\n        value {\n          func {\n            name: "while_cond_821"\n          }\n        }\n      }\n      attr {\n        key: "output_shapes"\n        value {\n          list {\n            shape {\n            }\n            shape {\n            }\n            shape {\n            }\n          }\n        }\n      }\n      attr {\n        key: "parallel_iterations"\n        value {\n          i: 10\n        }\n      }\n    }\n    node {\n      name: "while/Identity"\n      op: "Identity"\n      input: "while"\n      attr {\n        key: "T"\n        value {\n          type: DT_INT32\n        }\n      }\n    }\n    node {\n      name: "while/Identity_1"\n      op: "Identity"\n      input: "while:1"\n      attr {\n        key: "T"\n        value {\n          type: DT_INT32\n        }\n      }\n    }\n    node {\n      name: "while/Identity_2"\n      op: "Identity"\n      input: "while:2"\n      attr {\n        key: "T"\n        value {\n          type: DT_FLOAT\n        }\n      }\n    }\n    library {\n      function {\n        signature {\n          name: "while_body_822"\n          input_arg {\n            name: "while_loop_counter"\n            type: DT_INT32\n          }\n          input_arg {\n            name: "while_maximum_iterations_0"\n            type: DT_INT32\n          }\n          input_arg {\n            name: "placeholder"\n            type: DT_FLOAT\n          }\n          output_arg {\n            name: "add"\n            type: DT_INT32\n          }\n          output_arg {\n            name: "while_maximum_iterations"\n            type: DT_INT32\n          }\n          output_arg {\n            name: "partitionedcall"\n            type: DT_FLOAT\n          }\n        }\n        node_def {\n          name: "PartitionedCall"\n          op: "PartitionedCall"\n          input: "placeholder"\n          attr {\n            key: "Tin"\n            value {\n              list {\n                type: DT_FLOAT\n              }\n            }\n          }\n          attr {\n            key: "Tout"\n            value {\n              list {\n                type: DT_FLOAT\n              }\n            }\n          }\n          attr {\n            key: "_collective_manager_ids"\n            value {\n              list {\n              }\n            }\n          }\n          attr {\n            key: "_read_only_resource_inputs"\n            value {\n              list {\n              }\n            }\n          }\n          attr {\n            key: "config"\n            value {\n              s: ""\n            }\n          }\n          attr {\n            key: "config_proto"\n            value {\n              s: ""\n            }\n          }\n          attr {\n            key: "executor_type"\n            value {\n              s: ""\n            }\n          }\n          attr {\n            key: "f"\n            value {\n              func {\n                name: "__inference_f_841"\n              }\n            }\n          }\n          experimental_debug_info {\n            original_node_names: "PartitionedCall"\n          }\n        }\n        node_def {\n          name: "add/y"\n          op: "Const"\n          attr {\n            key: "dtype"\n            value {\n              type: DT_INT32\n            }\n          }\n          attr {\n            key: "value"\n            value {\n              tensor {\n                dtype: DT_INT32\n                tensor_shape {\n                }\n                int_val: 1\n              }\n            }\n          }\n          experimental_debug_info {\n            original_node_names: "add/y"\n          }\n        }\n        node_def {\n          name: "add_0"\n          op: "AddV2"\n          input: "while_loop_counter"\n          input: "add/y:output:0"\n          attr {\n            key: "T"\n            value {\n              type: DT_INT32\n            }\n          }\n          experimental_debug_info {\n            original_node_names: "add"\n          }\n        }\n        ret {\n          key: "add"\n          value: "add_0:z:0"\n        }\n        ret {\n          key: "partitionedcall"\n          value: "PartitionedCall:output:0"\n        }\n        ret {\n          key: "while_maximum_iterations"\n          value: "while_maximum_iterations_0"\n        }\n        arg_attr {\n          key: 0\n          value {\n            attr {\n              key: "_output_shapes"\n              value {\n                list {\n                  shape {\n                  }\n                }\n              }\n            }\n          }\n        }\n        arg_attr {\n          key: 1\n          value {\n            attr {\n              key: "_output_shapes"\n              value {\n                list {\n                  shape {\n                  }\n                }\n              }\n            }\n          }\n        }\n        arg_attr {\n          key: 2\n          value {\n            attr {\n              key: "_output_shapes"\n              value {\n                list {\n                  shape {\n                  }\n                }\n              }\n            }\n          }\n        }\n      }\n      function {\n        signature {\n          name: "while_cond_821"\n          input_arg {\n            name: "while_loop_counter"\n            type: DT_INT32\n          }\n          input_arg {\n            name: "while_maximum_iterations"\n            type: DT_INT32\n          }\n          input_arg {\n            name: "placeholder"\n            type: DT_FLOAT\n          }\n          output_arg {\n            name: "less"\n            type: DT_BOOL\n          }\n        }\n        node_def {\n          name: "Less/y"\n          op: "Const"\n          attr {\n            key: "dtype"\n            value {\n              type: DT_FLOAT\n            }\n          }\n          attr {\n            key: "value"\n            value {\n              tensor {\n                dtype: DT_FLOAT\n                tensor_shape {\n                }\n                float_val: 5.0\n              }\n            }\n          }\n          experimental_debug_info {\n            original_node_names: "Less/y"\n          }\n        }\n        node_def {\n          name: "Less"\n          op: "Less"\n          input: "placeholder"\n          input: "Less/y:output:0"\n          attr {\n            key: "T"\n            value {\n              type: DT_FLOAT\n            }\n          }\n          experimental_debug_info {\n            original_node_names: "Less"\n          }\n        }\n        ret {\n          key: "less"\n          value: "Less:z:0"\n        }\n        arg_attr {\n          key: 0\n          value {\n            attr {\n              key: "_output_shapes"\n              value {\n                list {\n                  shape {\n                  }\n                }\n              }\n            }\n          }\n        }\n        arg_attr {\n          key: 1\n          value {\n            attr {\n              key: "_output_shapes"\n              value {\n                list {\n                  shape {\n                  }\n                }\n              }\n            }\n          }\n        }\n        arg_attr {\n          key: 2\n          value {\n            attr {\n              key: "_output_shapes"\n              value {\n                list {\n                  shape {\n                  }\n                }\n              }\n            }\n          }\n        }\n      }\n      function {\n        signature {\n          name: "__inference_f_841"\n          input_arg {\n            name: "mul_placeholder"\n            type: DT_FLOAT\n          }\n          output_arg {\n            name: "identity"\n            type: DT_FLOAT\n          }\n        }\n        node_def {\n          name: "mul/y"\n          op: "Const"\n          attr {\n            key: "dtype"\n            value {\n              type: DT_FLOAT\n            }\n          }\n          attr {\n            key: "value"\n            value {\n              tensor {\n                dtype: DT_FLOAT\n                tensor_shape {\n                }\n                float_val: 2.0\n              }\n            }\n          }\n          experimental_debug_info {\n            original_node_names: "mul/y"\n          }\n        }\n        node_def {\n          name: "mul"\n          op: "Mul"\n          input: "mul_placeholder"\n          input: "mul/y:output:0"\n          attr {\n            key: "T"\n            value {\n              type: DT_FLOAT\n            }\n          }\n          experimental_debug_info {\n            original_node_names: "mul"\n          }\n        }\n        node_def {\n          name: "Identity"\n          op: "Identity"\n          input: "mul:z:0"\n          attr {\n            key: "T"\n            value {\n              type: DT_FLOAT\n            }\n          }\n          experimental_debug_info {\n            original_node_names: "Identity"\n          }\n        }\n        ret {\n          key: "identity"\n          value: "Identity:output:0"\n        }\n        arg_attr {\n          key: 0\n          value {\n            attr {\n              key: "_output_shapes"\n              value {\n                list {\n                  shape {\n                  }\n                }\n              }\n            }\n          }\n        }\n      }\n    }\n    versions {\n      producer: 399\n      min_consumer: 12\n    }\n    '
        graph_def = graph_pb2.GraphDef()
        text_format.Parse(serialized, graph_def)

        @def_function.function
        def F():
            if False:
                for i in range(10):
                    print('nop')
            (x, y) = importer.import_graph_def(graph_def, return_elements=['Const:0', 'while:2'])
            (grad_out,) = gradients_impl.gradients(y, x)
            return grad_out
        self.assertAllEqual(F(), 8.0)

    def testIndexedSlicesInIncomingGrads(self):
        if False:
            i = 10
            return i + 15

        @def_function.function
        def F():
            if False:
                i = 10
                return i + 15
            x = constant_op.constant([2.0])
            ret = while_loop_v2(lambda _: True, lambda v: v * v, [x], return_same_structure=False, maximum_iterations=2)
            v = array_ops.gather(ret, [0])
            return gradients_impl.gradients(v, [x])[0]
        self.assertAllEqual(self.evaluate(F()), [32.0])

    def testShapeInvariantsRaggedTensor(self):
        if False:
            print('Hello World!')

        @def_function.function
        def TestFn(x):
            if False:
                print('Hello World!')
            (_, ret) = while_loop_v2(lambda i, _: i < 1, lambda i, y: (i + 1, array_ops.concat([y, y], axis=0)), [0, x], shape_invariants=[tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32), ragged_tensor.RaggedTensorSpec(shape=[None, None])])
            return ret
        x = ragged_factory_ops.constant([[1.0, 2.0], [3.0]])
        result = TestFn(x)
        expected_result = [[1.0, 2.0], [3.0], [1.0, 2.0], [3.0]]
        self.assertAllEqual(result, expected_result)

def ScalarShape():
    if False:
        print('Hello World!')
    return ops.convert_to_tensor([], dtype=dtypes.int32)

def GetOptimizedGraph():
    if False:
        print('Hello World!')
    mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.CopyFrom(rewriter_config_pb2.RewriterConfig(constant_folding=rewriter_config_pb2.RewriterConfig.OFF, memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL))
    return tf_optimizer.OptimizeGraph(config, mg)
if __name__ == '__main__':
    test.main()