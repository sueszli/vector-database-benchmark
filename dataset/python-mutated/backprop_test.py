import functools
import sys
from absl.testing import parameterized
import numpy as np
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import record
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework.memory_checker import MemoryChecker
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_grad
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.training import training

class BackpropTest(test.TestCase, parameterized.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def testAggregateGradients(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(x):
            if False:
                i = 10
                return i + 15
            ind1 = constant_op.constant(np.array([0, 1]))
            ind2 = constant_op.constant(np.array([2, 3]))
            ind3 = constant_op.constant(np.array([1, 3]))
            g1 = embedding_ops.embedding_lookup(x, ind1)
            g2 = embedding_ops.embedding_lookup(x, ind2)
            g3 = embedding_ops.embedding_lookup(x, ind3)
            return g1 * g2 * g3
        var_np = np.random.rand(4, 2).astype(np.float32)
        var = constant_op.constant(var_np)
        grad = backprop.gradients_function(fn, [0])(var)[0]
        grad = self.evaluate(ops.convert_to_tensor(grad))
        if not context.executing_eagerly():
            tf_var = array_ops.constant(var_np, dtypes.float32)
            tf_ind1 = array_ops.constant([0, 1])
            tf_ind2 = array_ops.constant([2, 3])
            tf_ind3 = array_ops.constant([1, 3])
            tf_g1 = embedding_ops.embedding_lookup(tf_var, tf_ind1)
            tf_g2 = embedding_ops.embedding_lookup(tf_var, tf_ind2)
            tf_g3 = embedding_ops.embedding_lookup(tf_var, tf_ind3)
            tf_y = tf_g1 * tf_g2 * tf_g3
            tf_grad = gradients.gradients(tf_y, [tf_var])[0]
            tf_dense_grad = math_ops.unsorted_segment_sum(tf_grad.values, tf_grad.indices, tf_grad.dense_shape[0])
            self.assertAllClose(grad, self.evaluate(tf_dense_grad))

    @test_util.run_in_graph_and_eager_modes
    def testAggregateGradientsWithTensor(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            ind1 = constant_op.constant(np.array([0, 1]))
            g1 = embedding_ops.embedding_lookup(x, ind1)
            g2 = math_ops.reduce_sum(x * constant_op.constant(2.0))
            return g1 * g2
        var_np = np.random.rand(4, 2).astype(np.float32)
        var = constant_op.constant(var_np)
        grad = backprop.gradients_function(fn, [0])(var)[0]
        grad = self.evaluate(ops.convert_to_tensor(grad))
        if not context.executing_eagerly():
            tf_var = array_ops.constant(var_np, dtypes.float32)
            tf_ind1 = array_ops.constant([0, 1])
            tf_g1 = embedding_ops.embedding_lookup(tf_var, tf_ind1)
            tf_g2 = math_ops.reduce_sum(tf_var * 2.0, axis=(0, 1))
            tf_y = tf_g1 * tf_g2
            tf_grad = gradients.gradients(tf_y, [tf_var])[0]
            self.assertAllClose(grad, tf_grad)

    def testImplicitGradWithResourceVariable(self):
        if False:
            print('Hello World!')
        x = resource_variable_ops.ResourceVariable(initial_value=constant_op.constant(1.0), name='x')

        def fn():
            if False:
                while True:
                    i = 10
            b = constant_op.constant(2.0)
            c = math_ops.add(x.value(), b)
            return math_ops.add(c, constant_op.constant(3.0))
        grads_and_vars = backprop.implicit_grad(fn)()
        self.assertAllEqual(grads_and_vars[0][0], 1.0)
        self.assertAllEqual(id(grads_and_vars[0][1]), id(x))

    @parameterized.named_parameters([('Function', def_function.function), ('NoFunction', lambda f: f)])
    def testNoOpBehaviorConsistent(self, decorator):
        if False:
            print('Hello World!')

        @decorator
        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            x1 = array_ops.identity(x)
            x2 = math_ops.add_v2(x, 0)
            x3 = math_ops.subtract(x, 0)
            x4 = math_ops.multiply(x, 1)
            with backprop.GradientTape() as t:
                t.watch(x)
                t.watch(x1)
                t.watch(x2)
                t.watch(x3)
                t.watch(x4)
                y1 = x * 2.0
                y2 = x1 * 3.0
                y3 = x2 * 3.0
                y4 = x3 * 3.0
                y5 = x4 * 3.0
                loss = y1 + y2 + y3 + y4 + y5
            return t.gradient(loss, [x, x1, x2, x3, x4])
        self.assertAllClose([2.0, 3.0, 3.0, 3.0, 3.0], f(constant_op.constant(10.0)))

    def testResourceHandleOutputWithoutHandleData(self):
        if False:
            print('Hello World!')
        h = resource_variable_ops.var_handle_op(shape=[], dtype=dtypes.float32, shared_name='abc')
        with backprop.GradientTape() as tape:
            x = constant_op.constant(1.0)
            tape.watch(x)
            tape.watch(h)
            (y, h) = array_ops.identity_n([x, h])
        self.assertAllClose(1.0, tape.gradient(y, x))

    def testGradientInsideLoop(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():
            v = resource_variable_ops.ResourceVariable(1.0)

            def body(_):
                if False:
                    for i in range(10):
                        print('nop')
                _ = v + 1.0
                with backprop.GradientTape() as t:
                    result = v * 2
                self.assertIsNotNone(t.gradient(result, v))
                return 1.0
            while_loop.while_loop(lambda i: False, body, [1.0])

    def testWhereGradient(self):
        if False:
            return 10

        def f(x):
            if False:
                while True:
                    i = 10
            return array_ops.where(x < 10, x, x * x)
        g = backprop.gradients_function(f)
        self.assertAllEqual(g(5.0)[0], 1.0)
        self.assertAllEqual(g(50.0)[0], 100.0)

    def testTwoTargets(self):
        if False:
            return 10
        with backprop.GradientTape() as t:
            x = constant_op.constant(3.0)
            y = constant_op.constant(2.0)
            t.watch([x, y])
            xx = 2 * x
            yy = 3 * y
        (dx, dy) = t.gradient([xx, yy], [x, y])
        self.assertAllEqual(dx, 2.0)
        self.assertAllEqual(dy, 3.0)

    def testCustomGradientEmptyError(self):
        if False:
            return 10

        @custom_gradient.custom_gradient
        def identity(x):
            if False:
                i = 10
                return i + 15

            def grad(_):
                if False:
                    i = 10
                    return i + 15
                return []
            return (x, grad)
        x = variables.Variable(1.0)
        with backprop.GradientTape() as t:
            y = identity(x)
        with self.assertRaises(ValueError):
            t.gradient(y, [x])

    def test_stop_gradient_hides_downstream_ops(self):
        if False:
            return 10

        @custom_gradient.custom_gradient
        def _backward_pass_error(x):
            if False:
                return 10

            def _grad(_):
                if False:
                    print('Hello World!')
                raise AssertionError('Unexpectedly ran the backward function. This probably means that tf.GradientTape is not properly ignoring tensors downstream of tf.stop_gradient.')
            return (x, _grad)

        @def_function.function
        def f(x):
            if False:
                print('Hello World!')
            return _backward_pass_error(x)
        x = constant_op.constant(1.0)
        with backprop.GradientTape() as tape:
            tape.watch(x)
            y = f(array_ops.stop_gradient(x))
        self.assertIsNone(tape.gradient(y, x))

    def testOutputGradUsedInComputation(self):
        if False:
            return 10
        with backprop.GradientTape() as t:
            x = constant_op.constant(3.0)
            y = constant_op.constant(2.0)
            t.watch([x, y])
            loss = x * y
        (dx,) = t.gradient([loss, x], [x], output_gradients=[1.0, 2.0])
        self.assertAllEqual(dx, 4.0)

    def testDy(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                return 10
            return x
        grad_fn = backprop.gradients_function(f)
        self.assertAllEqual(2.0, grad_fn(1.0, dy=2.0)[0])

    def testGradientInteger(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + x
        int_tensor = constant_op.constant(1)
        self.assertEqual(backprop.gradients_function(f)(int_tensor)[0], None)

    def testErrors(self):
        if False:
            for i in range(10):
                print('nop')

        @custom_gradient.custom_gradient
        def f(x):
            if False:
                print('Hello World!')

            def grad(_):
                if False:
                    return 10
                raise RuntimeError('x')
            return (x, grad)
        with self.assertRaises(RuntimeError):
            backprop.gradients_function(f)(constant_op.constant(1.0))

    def testGradientsFunctionInCustomGradient(self):
        if False:
            while True:
                i = 10

        @custom_gradient.custom_gradient
        def f(x):
            if False:
                while True:
                    i = 10
            (y,) = backprop.gradients_function(lambda x: x * x)(x)

            def grad(dy):
                if False:
                    print('Hello World!')
                return [2 * dy]
            return (y, grad)
        self.assertAllEqual(f(1.0), 2.0)

    def testImplicitGradOverEmbeddingLookup(self):
        if False:
            while True:
                i = 10
        batch_size = 8
        embedding_size = 512
        vocab_size = 1000
        lrn_rate = 0.1
        random_init = random_ops.random_uniform([vocab_size, embedding_size])
        x = array_ops.ones(batch_size, dtypes.int64)
        embedding = resource_variable_ops.ResourceVariable(initial_value=random_init, dtype=dtypes.float32, name='embedding')

        def f():
            if False:
                for i in range(10):
                    print('nop')
            embedded_x = embedding_ops.embedding_lookup(embedding, x)
            return constant_op.constant(1.0, dtypes.float32) - embedded_x
        grad = backprop.implicit_grad(f)()[0][0]
        opt = training.GradientDescentOptimizer(lrn_rate)
        with ops.Graph().as_default(), self.cached_session():
            tf_x = array_ops.ones(batch_size, dtypes.int64)
            tf_embedding = variables.Variable(random_init.numpy(), name='tf_embedding')
            tf_embedded_x = embedding_ops.embedding_lookup(tf_embedding, tf_x)
            tf_y = 1.0 - tf_embedded_x
            tf_grad = gradients.gradients(tf_y, [tf_embedding])[0]
            tf_opt = training.GradientDescentOptimizer(0.1)
            tf_embedding.initializer.run()
            self.assertAllClose(tf_grad.indices, grad.indices)
            self.assertAllClose(tf_grad.values, grad.values)
            tf_opt.apply_gradients([(tf_grad, tf_embedding)]).run()
            expected = self.evaluate(tf_embedding)
        opt.apply_gradients([(grad, embedding)])
        self.assertAllClose(expected, embedding.read_value())

    def testImplicitGradOrdering(self):
        if False:
            i = 10
            return i + 15
        v0 = resource_variable_ops.ResourceVariable(1.0)
        v1 = resource_variable_ops.ResourceVariable(2.0)

        def f():
            if False:
                return 10
            x = v1 * v1
            y = v0 * v0
            return x + y
        grads = backprop.implicit_grad(f)()
        ordered_variables = [x[1] for x in grads]
        self.assertIs(ordered_variables[0], v0)
        self.assertIs(ordered_variables[1], v1)

    def testTapeNoOpGradient(self):
        if False:
            return 10
        x = constant_op.constant(3.0)
        with backprop.GradientTape() as t:
            t.watch(x)
            y = x
        self.assertEqual(t.gradient(y, x).numpy(), 1.0)

    def testTapeIdentityGradientIsIdentity(self):
        if False:
            while True:
                i = 10
        x = constant_op.constant(3.0)
        with backprop.GradientTape() as t:
            t.watch(x)
            y = array_ops.identity(x)
        self.assertEqual(t.gradient(y, x).numpy(), 1.0)

    def testFunctionIndexedSlicesGradient(self):
        if False:
            print('Hello World!')

        @def_function.function
        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + 1
        with backprop.GradientTape() as t:
            x = constant_op.constant([1.0])
            t.watch(x)
            y = f(x)
            y = array_ops.gather(y, [0])
        self.assertAllEqual(t.gradient(y, x), [1.0])

    def testTapeGradientMultiTargetOneIsSource(self):
        if False:
            i = 10
            return i + 15
        x = constant_op.constant(2.0)
        with backprop.GradientTape() as t:
            t.watch(x)
            y = x * x
        self.assertEqual(t.gradient([x, y], x).numpy(), 5.0)

    def testTapeNoOpGradientWithMultiTargetAllSource(self):
        if False:
            print('Hello World!')
        x = constant_op.constant(3.0)
        with backprop.GradientTape() as t:
            t.watch(x)
            y = x
        self.assertEqual(t.gradient([y, y], x).numpy(), 2.0)

    def testTapeNoOpGradientWithMultiTargetMultiSource(self):
        if False:
            for i in range(10):
                print('nop')
        x = constant_op.constant(3.0)
        y = constant_op.constant(5.0)
        with backprop.GradientTape() as t:
            t.watch(x)
            t.watch(y)
            z = y * y
        self.assertAllEqual(t.gradient([x, y, z], [x, y]), [1.0, 11.0])

    def testTapeGradientStringTarget(self):
        if False:
            i = 10
            return i + 15
        s = constant_op.constant('unknown', dtype=dtypes.string)
        x = constant_op.constant(3.0)
        with backprop.GradientTape() as t:
            t.watch(x)
            t.watch(s)
        grads = t.gradient(s, x)
        self.assertEqual(grads, None)

    def testTapeNoOpGradientStringSourceAndTarget(self):
        if False:
            while True:
                i = 10
        s = constant_op.constant('unknown', dtype=dtypes.string)
        with backprop.GradientTape() as t:
            t.watch(s)
        grads = t.gradient(s, s)
        self.assertEqual(grads, None)

    def testTapeNoOpGradientWithMultiTargetMultiSourceIncludeString(self):
        if False:
            i = 10
            return i + 15
        x = constant_op.constant(3.0)
        y = constant_op.constant(5.0)
        s = constant_op.constant('unknown', dtype=dtypes.string)
        with backprop.GradientTape() as t:
            t.watch(x)
            t.watch(y)
            t.watch(s)
            z = y * y
        grads = t.gradient([x, y, z, s], [x, y, s])
        self.assertAllEqual(grads[:2], [1.0, 11.0])
        self.assertEqual(grads[2], None)

    def testTapeNoOpOnVariableIsIdentity(self):
        if False:
            i = 10
            return i + 15
        v0 = resource_variable_ops.ResourceVariable(1.0)
        with backprop.GradientTape() as t:
            y = v0.read_value()
        self.assertEqual(t.gradient(y, v0).numpy(), 1.0)

    @test_util.assert_no_new_tensors
    @test_util.assert_no_garbage_created
    def testTapeNoOpGradient2By2(self):
        if False:
            for i in range(10):
                print('nop')
        a_2_by_2 = constant_op.constant(2.0, shape=[2, 2])
        with backprop.GradientTape(persistent=True) as tape:
            tape.watch(a_2_by_2)
        dy_dy = tape.gradient(a_2_by_2, [a_2_by_2])[0]
        self.assertAllEqual(dy_dy.numpy(), constant_op.constant(1.0, shape=[2, 2]).numpy())

    @test_util.assert_no_new_pyobjects_executing_eagerly
    def testTapeNoOpGradientMultiTarget2By2(self):
        if False:
            while True:
                i = 10
        a_2_by_2 = constant_op.constant(2.0, shape=[2, 2])
        with backprop.GradientTape(persistent=True) as tape:
            tape.watch(a_2_by_2)
        dy_dy = tape.gradient([a_2_by_2, a_2_by_2], [a_2_by_2])[0]
        self.assertAllEqual(dy_dy.numpy(), constant_op.constant(2.0, shape=[2, 2]).numpy())

    def testTapeStopRecording(self):
        if False:
            i = 10
            return i + 15
        with backprop.GradientTape() as t:
            x = resource_variable_ops.ResourceVariable(1.0)
            with t.stop_recording():
                y = x * x
        self.assertEqual(t.gradient(y, x), None)

    def testTapeStopStartRecording(self):
        if False:
            while True:
                i = 10
        with backprop.GradientTape(persistent=True) as t:
            x = resource_variable_ops.ResourceVariable(1.0)
            x2 = x * 2
            with t.stop_recording():
                y = x2 * x2
            z = x2 * x2
        self.assertEqual(t.gradient(y, x2), None)
        self.assertEqual(t.gradient(z, x2).numpy(), 4.0)

    def testTapeReset(self):
        if False:
            print('Hello World!')
        with backprop.GradientTape() as t:
            v = resource_variable_ops.ResourceVariable(1.0)
            loss = v * v
            t.reset()
            loss += v * v
        self.assertAllEqual(t.gradient(loss, v), 2.0)

    def testPythonMax(self):
        if False:
            print('Hello World!')
        x = [resource_variable_ops.ResourceVariable(2.0), resource_variable_ops.ResourceVariable(3.0), resource_variable_ops.ResourceVariable(5.0)]
        with backprop.GradientTape() as t:
            f = max(x)
        grad = t.gradient(f, x)
        self.assertAllEqual(self.evaluate(f), 5.0)
        self.assertAllEqual(self.evaluate(grad), [None, None, 1.0])

    def testAutomaticWatchedVariables(self):
        if False:
            return 10
        with backprop.GradientTape() as t:
            self.assertEqual(0, len(t.watched_variables()))
            v = resource_variable_ops.ResourceVariable(1.0)
            loss = v * v
            self.assertAllEqual([v], t.watched_variables())
            t.reset()
            self.assertEqual(0, len(t.watched_variables()))
            loss += v * v
            self.assertAllEqual([v], t.watched_variables())

    def testExplicitWatchedVariables(self):
        if False:
            i = 10
            return i + 15
        with backprop.GradientTape() as t:
            self.assertEqual(0, len(t.watched_variables()))
            v = resource_variable_ops.ResourceVariable(1.0)
            t.watch(v)
            self.assertAllEqual([v], t.watched_variables())
            t.reset()
            self.assertEqual(0, len(t.watched_variables()))
            t.watch(v)
            self.assertAllEqual([v], t.watched_variables())

    @test_util.assert_no_new_tensors
    def testGradientNone(self):
        if False:
            print('Hello World!')

        def loss(x, l):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.reduce_mean(nn_ops.softmax_cross_entropy_with_logits(logits=x, labels=l), constant_op.constant([0]))
        logits = constant_op.constant([[0.0, 0.0]])
        labels = constant_op.constant([[1.0, 0.0]])
        (g,) = backprop.gradients_function(loss, [0])(logits, labels)
        self.assertAllEqual(g.numpy(), [[-0.5, 0.5]])

    @test_util.run_in_graph_and_eager_modes
    def testGradientWithinTapeBlock(self):
        if False:
            while True:
                i = 10
        v1 = resource_variable_ops.ResourceVariable(1.0)
        self.evaluate(v1.initializer)
        with backprop.GradientTape() as t:
            loss = 2 * v1
            grad = t.gradient(loss, v1)
        self.assertAllEqual(self.evaluate(grad), 2.0)
        with backprop.GradientTape(persistent=True) as t:
            loss = 2 * v1
            grad = t.gradient(loss, v1)
        self.assertAllEqual(self.evaluate(grad), 2.0)

    @test_util.run_in_graph_and_eager_modes
    def testNestedSelfContexts(self):
        if False:
            print('Hello World!')
        v1 = resource_variable_ops.ResourceVariable(1.0)
        self.evaluate(v1.initializer)
        with backprop.GradientTape() as t:
            with self.assertRaises(ValueError):
                with t:
                    pass

    @test_util.assert_no_new_tensors
    def testSecondGrad(self):
        if False:
            for i in range(10):
                print('nop')

        def first(x):
            if False:
                i = 10
                return i + 15
            l = constant_op.constant([[0.0]])
            x = nn_ops.softmax_cross_entropy_with_logits(labels=l, logits=x)
            x = math_ops.reduce_sum(x, constant_op.constant([0]))
            return x

        def second(x):
            if False:
                i = 10
                return i + 15
            grad = backprop.gradients_function(first, [0])(x)[0]
            return math_ops.reduce_sum(grad, constant_op.constant([0]))
        f = constant_op.constant([[0.1]])
        grad = backprop.gradients_function(second, [0])(f)[0]
        self.assertAllEqual([[0.0]], grad)

    @test_util.run_in_graph_and_eager_modes
    def testWatchingIsTapeLocal(self):
        if False:
            print('Hello World!')
        x1 = resource_variable_ops.ResourceVariable(2.0, trainable=False)
        x2 = resource_variable_ops.ResourceVariable(2.0, trainable=False)
        with backprop.GradientTape() as tape1:
            with backprop.GradientTape() as tape2:
                tape1.watch(x1)
                tape2.watch([x1, x2])
                y = x1 ** 3
                z = x2 ** 2
                (dy, dz) = tape2.gradient([y, z], [x1, x2])
            (d2y, d2z) = tape1.gradient([dy, dz], [x1, x2])
        self.evaluate([x1.initializer, x2.initializer])
        self.assertEqual(self.evaluate(d2y), 12.0)
        self.assertIsNone(d2z)

    @test_util.assert_no_new_tensors
    def testMakeVJP(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x * x
        wrapped_fn = backprop.make_vjp(f, persistent=False)
        (result, vjp) = wrapped_fn(constant_op.constant(3.0))
        self.assertAllEqual(result, 9.0)
        self.assertAllEqual(vjp(2.0)[0], 12.0)

    def testPersistentMakeVJP(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                return 10
            return x * x
        wrapped_fn = backprop.make_vjp(f, persistent=True)
        (_, vjp) = wrapped_fn(constant_op.constant(3.0))
        vjp_result1 = vjp(2.0)[0]
        vjp_result2 = vjp(2.0)[0]
        self.assertAllEqual(vjp_result1, vjp_result2, 12.0)

    @test_util.assert_no_new_tensors
    def testGradGrad(self):
        if False:
            i = 10
            return i + 15

        def sq(x):
            if False:
                for i in range(10):
                    print('nop')
            return x * x

        def grad(x):
            if False:
                print('Hello World!')
            value = backprop.gradients_function(sq, [0])(x)[0]
            return value
        gradgrad = backprop.gradients_function(grad, [0])
        self.assertAllEqual(gradgrad(constant_op.constant(3.0))[0], 2.0)

    @test_util.assert_no_new_tensors
    def testGradGradExp(self):
        if False:
            return 10

        def grad(x):
            if False:
                while True:
                    i = 10
            value = backprop.gradients_function(math_ops.exp, [0])(x)[0]
            return value
        gradgrad = backprop.gradients_function(grad, [0])
        self.assertAllEqual(gradgrad(constant_op.constant(0.0))[0], 1.0)

    @test_util.assert_no_new_tensors
    def testStopGradient(self):
        if False:
            for i in range(10):
                print('nop')
        grad = backprop.gradients_function(lambda x: array_ops.stop_gradient(math_ops.argmax(x)))
        self.assertAllEqual(grad([0.0])[0], None)

    @test_util.assert_no_new_tensors
    def testArgmax(self):
        if False:
            while True:
                i = 10

        def argmax(x):
            if False:
                while True:
                    i = 10
            i = math_ops.argmax(x)
            return array_ops.stop_gradient(i)
        grad = backprop.gradients_function(argmax)
        self.assertAllEqual(grad([0.0])[0], None)

    @test_util.run_gpu_only
    @test_util.assert_no_new_tensors
    def testGPU(self):
        if False:
            print('Hello World!')

        def fn(x):
            if False:
                print('Hello World!')
            with context.device('/gpu:0'):
                b = constant_op.constant(2.0)
                c = math_ops.add(x.gpu(), b)
                return math_ops.add(c, constant_op.constant(3.0)).cpu()
        grad = backprop.gradients_function(fn, [0])(constant_op.constant(1.0))[0]
        self.assertAllEqual(grad, 1.0)

    @test_util.run_gpu_only
    @test_util.assert_no_new_tensors
    def testGPUImplicitGrad(self):
        if False:
            i = 10
            return i + 15
        with context.device('gpu:0'):
            v = resource_variable_ops.ResourceVariable(constant_op.constant(1.0), name='v')

        def f():
            if False:
                return 10
            with context.device('gpu:0'):
                return v.read_value()
        self.assertEqual(backprop.implicit_grad(f)()[0][0].cpu().numpy(), 1.0)

    @test_util.assert_no_new_tensors
    def testCPU(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            b = constant_op.constant(2.0)
            c = math_ops.add(x, b)
            return math_ops.add(c, constant_op.constant(3.0))
        grad = backprop.gradients_function(fn, [0])(constant_op.constant(1.0))[0]
        self.assertAllEqual(grad, 1.0)

    @test_util.run_gpu_only
    @test_util.assert_no_new_tensors
    def testTensorCopyGPU2CPU2GPU(self):
        if False:
            i = 10
            return i + 15

        def f(a, b):
            if False:
                print('Hello World!')
            return a.cpu() + b.cpu()
        with context.device('/gpu:0'):
            a = constant_op.constant(1.0)
            b = constant_op.constant(2.0)
        grad = backprop.gradients_function(f, [0])(a, b)[0]
        self.assertAllEqual(grad, 1.0)

    @test_util.assert_no_new_tensors
    def testEmptyParams(self):
        if False:
            print('Hello World!')

        def fn(a, b):
            if False:
                i = 10
                return i + 15
            return a * b
        x = constant_op.constant(1.0)
        y = constant_op.constant(2.0)
        (dx, dy) = backprop.gradients_function(fn)(x, y)
        self.assertAllEqual(dx, y.numpy())
        self.assertAllEqual(dy, x.numpy())

    @test_util.assert_no_new_tensors
    def testUnconnectedNone(self):
        if False:
            for i in range(10):
                print('nop')
        v = resource_variable_ops.ResourceVariable(1.0, name='testUnconnectedNone')

        def f():
            if False:
                print('Hello World!')
            v.read_value()
            return constant_op.constant(1.0)
        self.assertEqual(backprop.implicit_grad(f)()[0][0], None)

    @test_util.assert_no_new_tensors
    def testGradientTapeReEnterContext(self):
        if False:
            print('Hello World!')
        g = backprop.GradientTape()
        with g:
            x = constant_op.constant(3.0)
            g.watch(x)
            y = 2 * x
        with g:
            z = 2 * y
        grad = g.gradient(target=z, sources=[x])
        self.assertEqual(self.evaluate(grad), [4.0])

    @test_util.assert_no_new_tensors
    @test_util.run_in_graph_and_eager_modes
    def testGradientTapeRepeatedSource(self):
        if False:
            while True:
                i = 10
        with backprop.GradientTape(persistent=False) as g:
            x = constant_op.constant(3.0)
            g.watch(x)
            y = 2 * x
        grad = g.gradient(target=y, sources=[x, x])
        self.assertEqual(self.evaluate(grad), [2.0, 2.0])

    @test_util.assert_no_new_tensors
    @test_util.run_in_graph_and_eager_modes
    def testPersistentGradientTapeRepeatedSource(self):
        if False:
            print('Hello World!')
        with backprop.GradientTape(persistent=True) as g:
            x = constant_op.constant(3.0)
            y = constant_op.constant(5.0)
            g.watch(x)
            g.watch(y)
            z = x * x + x * y
        grad = g.gradient(target=z, sources=[x, x])
        self.assertEqual(self.evaluate(grad), [11.0, 11.0])
        grad = g.gradient(target=z, sources=[y, x])
        self.assertEqual(self.evaluate(grad), [3.0, 11.0])

    @test_util.assert_no_new_tensors
    @test_util.run_in_graph_and_eager_modes
    def testGradientTapeStructure(self):
        if False:
            for i in range(10):
                print('nop')
        with backprop.GradientTape(persistent=True) as g:
            x1 = constant_op.constant(3.0)
            x2 = constant_op.constant(3.1)
            x3 = constant_op.constant(3.2)
            g.watch(x1)
            g.watch(x2)
            g.watch(x3)
            y = x1 + 2 * x2 + 3 * x3
        self.assertEqual(self.evaluate(g.gradient(y, x1)), [1.0])
        self.assertEqual(self.evaluate(g.gradient(y, (x1,))), (1.0,))
        self.assertEqual(self.evaluate(g.gradient(y, (x1, x2))), (1.0, 2.0))
        self.assertEqual(self.evaluate(g.gradient(y, [(x1, x2), (x2, x3)])), [(1.0, 2.0), (2.0, 3.0)])
        self.assertEqual(self.evaluate(g.gradient(y, (x1, x2, [x1, x3]))), (1.0, 2.0, [1.0, 3.0]))
        self.assertEqual(self.evaluate(g.gradient(y, [x1, {'x2': x2, 'x3': x3}])), [1.0, {'x2': 2.0, 'x3': 3.0}])

    @test_util.assert_no_new_tensors
    @test_util.run_in_graph_and_eager_modes
    def testGradientTape(self):
        if False:
            while True:
                i = 10
        with backprop.GradientTape() as g:
            x = constant_op.constant(3.0)
            g.watch(x)
            y = x * x
            with backprop.GradientTape() as gg:
                gg.watch(y)
                z = 2 * y
            inner_grad = gg.gradient(z, [y])[0]
            self.assertEqual(self.evaluate(inner_grad), 2.0)
            y += inner_grad
        grad = g.gradient(y, [x])[0]
        self.assertEqual(self.evaluate(grad), 6.0)

    @test_util.assert_no_new_tensors
    @test_util.run_in_graph_and_eager_modes
    def testGadientTapeCalledOnConstantTarget(self):
        if False:
            print('Hello World!')
        with backprop.GradientTape() as g:
            x = variables.Variable([3.0])
            y = variables.Variable([2.0])
        grad = g.gradient(x, y)
        self.assertAllEqual(grad, None)

    @test_util.run_in_graph_and_eager_modes
    @test_util.run_v1_only('b/120545219')
    def testGradientTapeWithCond(self):
        if False:
            return 10
        x = constant_op.constant(3.0)

        def true_fn():
            if False:
                for i in range(10):
                    print('nop')
            return x

        def false_fn():
            if False:
                while True:
                    i = 10
            return x * x
        with backprop.GradientTape() as g:
            g.watch(x)
            y = tf_cond.cond(x < x, true_fn, false_fn)
        if not context.executing_eagerly():
            with self.assertRaisesRegex(NotImplementedError, 'tf.gradients'):
                dy = g.gradient(y, [x])[0]
        else:
            dy = g.gradient(y, [x])[0]
            self.assertEqual(self.evaluate(dy), 6.0)

    @test_util.run_in_graph_and_eager_modes
    @test_util.run_v1_only('b/120545219')
    def testGradientTapeWithWhileLoop(self):
        if False:
            i = 10
            return i + 15
        i = constant_op.constant(1)
        x = constant_op.constant(2.0)

        def cond(i, _):
            if False:
                for i in range(10):
                    print('nop')
            return i < 3

        def body(i, x):
            if False:
                for i in range(10):
                    print('nop')
            return (i + 1, x * 2)
        with backprop.GradientTape() as g:
            g.watch([x])
            (_, y) = while_loop.while_loop(cond, body, [i, x])
        if not context.executing_eagerly():
            with self.assertRaisesRegex(NotImplementedError, 'tf.gradients'):
                dy = g.gradient(y, [x])[0]
        else:
            dy = g.gradient(y, [x])[0]
            self.assertEqual(self.evaluate(dy), 4.0)

    @test_util.assert_no_new_tensors
    def testGradientTapeGradientCalledMultipleTimes(self):
        if False:
            print('Hello World!')
        with backprop.GradientTape() as g:
            x = constant_op.constant(3.0)
            g.watch(x)
            y = x * x
            z = y * y
        g.gradient(z, [x])
        with self.assertRaisesRegex(RuntimeError, 'A non-persistent GradientTape can only'):
            g.gradient(y, [x])

    @test_util.assert_no_new_tensors
    def testGradientTapeJacobianCalledMultipleTimes(self):
        if False:
            i = 10
            return i + 15
        with backprop.GradientTape() as g:
            x = constant_op.constant(3.0)
            g.watch(x)
            y = x * x
            z = y * y
        g.jacobian(z, [x])
        with self.assertRaisesRegex(RuntimeError, 'A non-persistent GradientTape can only'):
            g.jacobian(y, [x])

    @test_util.assert_no_new_tensors
    def testJacobianInsideGradientTapeScope(self):
        if False:
            for i in range(10):
                print('nop')
        with backprop.GradientTape() as g:
            x = constant_op.constant(3.0)
            g.watch(x)
            y = x * x
            z = y * y
            self.assertAllClose(4.0 * 3.0 ** 3.0, g.jacobian(z, x))

    @test_util.assert_no_new_tensors
    def testBatchJacobianInsideGradientTapeScope(self):
        if False:
            for i in range(10):
                print('nop')
        with backprop.GradientTape(persistent=True) as g:
            x = constant_op.constant([[3.0]])
            g.watch(x)
            y = x * x
            z = y * y
            self.assertAllClose([[[4.0 * 3.0 ** 3.0]]], g.batch_jacobian(z, x))

    def testBatchJacobianParallelIterations(self):
        if False:
            for i in range(10):
                print('nop')

        @def_function.function
        def f(persistent):
            if False:
                return 10
            with backprop.GradientTape(persistent=persistent) as t:
                x = constant_op.constant([[3.0]])
                t.watch(x)
                y = x * x
                z = array_ops.tile(y * y, [1, 16])
            return t.batch_jacobian(z, x, parallel_iterations=8)
        with self.assertRaisesRegex(RuntimeError, 'persistent=True.*parallel_iterations'):
            f(persistent=False)
        self.assertAllClose([[[4.0 * 3.0 ** 3.0]] * 16], f(persistent=True))

    @test_util.assert_no_new_tensors
    def testGradientTapeBatchJacobianCalledMultipleTimes(self):
        if False:
            while True:
                i = 10
        with backprop.GradientTape() as g:
            x = constant_op.constant([[3.0]])
            g.watch(x)
            y = x * x
            z = y * y
        g.batch_jacobian(z, x)
        with self.assertRaisesRegex(RuntimeError, 'A non-persistent GradientTape can only'):
            g.batch_jacobian(y, [x])

    @test_util.assert_no_new_tensors
    @test_util.run_in_graph_and_eager_modes
    @test_util.run_v1_only('b/120545219')
    def testPersistentTape(self):
        if False:
            for i in range(10):
                print('nop')
        with backprop.GradientTape(persistent=True) as g:
            x = constant_op.constant(3.0)
            g.watch(x)
            y = x * x
            z = y * y
        dz_dx = g.gradient(z, [x])[0]
        self.assertEqual(self.evaluate(dz_dx), 4 * 3 * 3 * 3)
        dy_dx = g.gradient(y, [x])[0]
        self.assertEqual(self.evaluate(dy_dx), 2 * 3)
        del g

    @test_util.assert_no_new_tensors
    @test_util.run_in_graph_and_eager_modes
    def testHigherOrderGradient(self):
        if False:
            print('Hello World!')
        with backprop.GradientTape(persistent=True) as g:
            x = constant_op.constant(3.0)
            g.watch(x)
            y = x ** 3
            dy_dx = g.gradient(y, x)
            d2y_dx2 = g.gradient(dy_dx, x)
        d3y_dx3 = g.gradient(d2y_dx2, x)
        x = 3
        self.assertAllClose(self.evaluate(y), x ** 3)
        self.assertEqual(self.evaluate(dy_dx), 3 * x ** 2)
        self.assertEqual(self.evaluate(d2y_dx2), 6 * x)
        self.assertEqual(self.evaluate(d3y_dx3), 6)
        del g

    @test_util.assert_no_new_tensors
    @test_util.run_in_graph_and_eager_modes
    def testPersistentNestedTape(self):
        if False:
            i = 10
            return i + 15
        with backprop.GradientTape(persistent=True) as g:
            x = constant_op.constant(3.0)
            g.watch(x)
            y = x * x
            with backprop.GradientTape(persistent=True) as gg:
                gg.watch(y)
                z = 2 * y
            for _ in range(2):
                inner_grad = gg.gradient(z, [y])[0]
                self.assertEqual(self.evaluate(inner_grad), 2.0)
            y += inner_grad
            del gg
        grad = g.gradient(y, [x])[0]
        self.assertEqual(self.evaluate(grad), 6.0)
        grad = g.gradient(z, [x])[0]
        self.assertEqual(self.evaluate(grad), 12.0)
        del g

    @test_util.assert_no_new_tensors
    @test_util.run_in_graph_and_eager_modes
    def testGradientTapeVariable(self):
        if False:
            print('Hello World!')
        v = resource_variable_ops.ResourceVariable(1.0, name='v')
        self.evaluate(v.initializer)
        with backprop.GradientTape() as g:
            y = v * v
        grad = g.gradient(y, [v])[0]
        self.assertAllEqual(self.evaluate(grad), 2.0)

    @test_util.assert_no_new_tensors
    @test_util.run_in_graph_and_eager_modes
    def testNestedGradients(self):
        if False:
            i = 10
            return i + 15
        x = constant_op.constant(3.0)
        with backprop.GradientTape() as g:
            g.watch(x)
            y = x * x
            z = y * y
        (dz_dx, dz_dy) = g.gradient(z, [x, y])
        self.assertEqual(self.evaluate(dz_dx), 108.0)
        self.assertEqual(self.evaluate(dz_dy), 18.0)

    @test_util.assert_no_new_tensors
    @test_util.run_in_graph_and_eager_modes
    def testUnconnectedGradientsDefault(self):
        if False:
            for i in range(10):
                print('nop')
        x = constant_op.constant(1.0)
        y = constant_op.constant(3.0)
        with backprop.GradientTape() as g:
            g.watch([x, y])
            z = y * 2
        dz_dx = g.gradient(z, x)
        self.assertEqual(dz_dx, None)

    @test_util.assert_no_new_tensors
    @test_util.run_in_graph_and_eager_modes
    def testUnconnectedGradientsZeros(self):
        if False:
            print('Hello World!')
        x = constant_op.constant(1.0, shape=[2, 2])
        y = constant_op.constant(3.0)
        with backprop.GradientTape() as g:
            g.watch([x, y])
            z = y * 2
        dz_dx = g.gradient(z, x, unconnected_gradients='zero')
        self.assertAllEqual([[0.0, 0.0], [0.0, 0.0]], self.evaluate(dz_dx))

    @test_util.assert_no_new_tensors
    @test_util.run_in_graph_and_eager_modes
    def testUnconnectedGradientsVariablesZeros(self):
        if False:
            i = 10
            return i + 15
        x = resource_variable_ops.ResourceVariable(constant_op.constant(1.0, shape=[2, 2]))
        self.evaluate(x.initializer)
        y = resource_variable_ops.ResourceVariable(constant_op.constant(3.0))
        self.evaluate(y.initializer)
        with backprop.GradientTape() as g:
            g.watch([x, y])
            z = y * 2
        dz_dx = g.gradient(z, x, unconnected_gradients='zero')
        self.assertAllEqual([[0.0, 0.0], [0.0, 0.0]], self.evaluate(dz_dx))

    @test_util.run_in_graph_and_eager_modes
    def testUnknownUnconnectedGradientsValueGiven(self):
        if False:
            while True:
                i = 10
        x = constant_op.constant(1.0)
        y = constant_op.constant(1.0)
        with backprop.GradientTape() as g:
            g.watch([x, y])
            z = y * 2
        with self.assertRaisesRegex(ValueError, "Unknown value for unconnected_gradients: 'nonsense'"):
            g.gradient(z, x, unconnected_gradients='nonsense')

    @test_util.run_in_graph_and_eager_modes
    def testUnconnectedGradientsNestedDefunZeros(self):
        if False:
            return 10

        @def_function.function
        def f(x):
            if False:
                while True:
                    i = 10
            return x * x

        @def_function.function
        def h(y):
            if False:
                print('Hello World!')
            z = f(y)
            return array_ops.stop_gradient(z)
        x = constant_op.constant(1.0)
        with backprop.GradientTape() as g:
            g.watch(x)
            k = x + 2.0
            y = h(k)
        dy_dx = g.gradient(y, x, unconnected_gradients='zero')
        self.assertEqual(0.0, self.evaluate(dy_dx))

    def testInvalidRecordOperationMessage(self):
        if False:
            i = 10
            return i + 15
        y = constant_op.constant(2.0)
        x = constant_op.constant(1.0)
        with backprop.GradientTape() as g:
            g.watch(x)
            record.record_operation('InvalidBackprop', [y], [x], lambda dy: [])
        with self.assertRaisesRegex(errors_impl.InternalError, 'InvalidBackprop.*too few gradients'):
            g.gradient(y, x)

    @test_util.assert_no_new_tensors
    def testEmptyParamsForValueAndGradFunction(self):
        if False:
            return 10

        def fn(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return a * b
        val_and_grads_fn = backprop.val_and_grad_function(fn)
        x = 2.0
        y = 3.0
        (val, (dx, dy)) = val_and_grads_fn(x, y)
        self.assertAllClose(val, x * y)
        self.assertAllEqual(dx, y)
        self.assertAllEqual(dy, x)

    @test_util.assert_no_new_tensors
    def testNonEmptyParamsForValueAndGradFunction(self):
        if False:
            while True:
                i = 10

        def fn(a, b):
            if False:
                i = 10
                return i + 15
            return a * b
        val_and_grad_fn = backprop.val_and_grad_function(fn, params=[1])
        x = 2.0
        y = 3.0
        (val, grads) = val_and_grad_fn(x, y)
        self.assertAllClose(val, x * y)
        self.assertEqual(1, len(grads))
        self.assertAllEqual(grads[0], x)

    @test_util.run_gpu_only
    @test_util.assert_no_new_tensors
    def testTensorCopyCPU2GPU2CPU(self):
        if False:
            print('Hello World!')

        def f(a, b):
            if False:
                while True:
                    i = 10
            with context.device('/gpu:0'):
                c = math_ops.add(a.gpu(0), b.gpu(0))
            return math_ops.add(c.cpu(), constant_op.constant(3.0))
        with context.device('/cpu:0'):
            a = constant_op.constant(1.0)
            b = constant_op.constant(2.0)
        grad = backprop.gradients_function(f, [0])(a, b)[0]
        self.assertAllEqual(grad, 1.0)

    def testGetAttrType(self):
        if False:
            while True:
                i = 10
        typ = backprop.op_attr_type('Add', 'T')
        self.assertEqual(typ, int(pywrap_tfe.TF_ATTR_TYPE))

    def testGetAttrList(self):
        if False:
            print('Hello World!')
        typ = backprop.op_attr_type('MaxPool', 'ksize')
        self.assertEqual(typ, [int(pywrap_tfe.TF_ATTR_INT)])

    def testMakeAttrType(self):
        if False:
            while True:
                i = 10
        self.assertEqual(dtypes.float32, backprop.make_attr(int(pywrap_tfe.TF_ATTR_TYPE), 1))

    def testMakeAttrTypeList(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual([dtypes.float32], backprop.make_attr([int(pywrap_tfe.TF_ATTR_TYPE)], [1]))

    def testMakeAttrString(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(b'a', backprop.make_attr(int(pywrap_tfe.TF_ATTR_STRING), 'a'))

    def testMakeAttrStringList(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual([b'a'], backprop.make_attr([int(pywrap_tfe.TF_ATTR_STRING)], ['a']))

    def testMulType(self):
        if False:
            return 10

        def mul(x):
            if False:
                print('Hello World!')
            return math_ops._mul_dispatch(x, x)
        self.assertAllEqual(backprop.gradients_function(mul)(3.0)[0].numpy(), 6.0)

    def testMakeAttrShape(self):
        if False:
            while True:
                i = 10
        for s in ([], None, [1, 2, 3], [None, None], [1, None, 3]):
            expected = tensor_shape.TensorShape(s).as_proto()
            actual = backprop.make_attr(int(pywrap_tfe.TF_ATTR_SHAPE), s)
            self.assertEqual(expected, actual, msg='For shape %r, expected %r != %r actual' % (s, expected, actual))

    def testMakeAttrShapeList(self):
        if False:
            while True:
                i = 10
        shape_list = [[], None, [1, 2, 3], [None, None], [1, None, 3]]
        self.assertEqual([tensor_shape.TensorShape(s).as_proto() for s in shape_list], backprop.make_attr([int(pywrap_tfe.TF_ATTR_SHAPE)], shape_list))

    def testArgsGradientFunction(self):
        if False:
            i = 10
            return i + 15

        def f(*args):
            if False:
                for i in range(10):
                    print('nop')
            return args[0] * args[0]
        grad = backprop.gradients_function(f)
        self.assertAllEqual(grad(1.0)[0], 2.0)

    def testPartial(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x * y
        part = functools.partial(f, constant_op.constant(2.0))
        self.assertAllEqual(backprop.gradients_function(part)(constant_op.constant(1.0))[0], 2.0)

    def testReturnSameThing(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                return 10
            return (x, 2 * x)
        self.assertAllEqual(backprop.gradients_function(f)(1.0)[0], 3.0)

    @test_util.assert_no_new_tensors
    def testExceptionSafety(self):
        if False:
            for i in range(10):
                print('nop')

        def f(unused_x):
            if False:
                i = 10
                return i + 15
            raise ValueError()
        try:
            backprop.gradients_function(f)(1.0)
        except ValueError:
            pass

        def real_f(x):
            if False:
                while True:
                    i = 10
            return x * x
        self.assertAllEqual(backprop.gradients_function(real_f)(1.0)[0], 2.0)

    @test_util.assert_no_new_tensors
    def testMultiValueConvertToTensor(self):
        if False:
            print('Hello World!')
        x = resource_variable_ops.ResourceVariable(initial_value=array_ops.constant([1.0]), name='x')

        def fn():
            if False:
                print('Hello World!')
            a = math_ops.add(x.value(), 1.0)
            b = array_ops_stack.stack([a, a], axis=0)
            return math_ops.reduce_mean(b)
        grad = backprop.implicit_grad(fn)()[0][0]
        self.assertAllEqual([1.0], grad)

    def testOutput(self):
        if False:
            while True:
                i = 10

        def multiout(x):
            if False:
                for i in range(10):
                    print('nop')
            return (x + 2, x * x)
        x = constant_op.constant([0.0, 1.0, 2.0])
        grad = backprop.gradients_function(multiout)(x)[0]
        self.assertAllEqual([1.0, 3.0, 5.0], grad)

    def testMultiValuePreservesIfNotDiffedAgainst(self):
        if False:
            while True:
                i = 10

        def tfe_conv2d(timage, tkernel, conv2dstrides):
            if False:
                print('Hello World!')
            return nn_ops.conv2d(timage, tkernel, conv2dstrides, 'SAME')
        i = constant_op.constant([[[[1.0]]]])
        k = constant_op.constant([[[[2.0]]]])
        s = [1, 1, 1, 1]
        grad = backprop.gradients_function(tfe_conv2d, params=(0,))(i, k, s)[0]
        self.assertAllEqual([[[[2.0]]]], grad)

    def testSameObjectForMultipleArguments(self):
        if False:
            i = 10
            return i + 15

        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.multiply(x, y)
        g = backprop.gradients_function(f)

        def np_g(x, y):
            if False:
                while True:
                    i = 10
            (dx, dy) = g(x, y)
            return [dx.numpy(), dy.numpy()]
        x = constant_op.constant(1.0)
        self.assertAllEqual([1.0, 1.0], np_g(x, x))
        x = 1.0
        self.assertAllEqual([1.0, 1.0], np_g(x, x))
        x = constant_op.constant([[1.0]])
        self.assertAllEqual([[[1.0]], [[1.0]]], np_g(x, x))
        x = [[1.0]]
        self.assertAllEqual([[[1.0]], [[1.0]]], np_g(x, x))
        v = resource_variable_ops.ResourceVariable(initial_value=1.0, name='testSameObjectForMultipleArguments.Variable')
        self.assertAllEqual([1.0, 1.0], np_g(v, v))

    @test_util.assert_no_new_tensors
    def testImplicitGradientsCustomGradientAndCachedVariableValue(self):
        if False:
            print('Hello World!')

        @custom_gradient.custom_gradient
        def my_square(x):
            if False:
                for i in range(10):
                    print('nop')
            result = math_ops.square(x)

            def grad(dr):
                if False:
                    while True:
                        i = 10
                return 2 * dr * x + 1
            return (result, grad)
        x = resource_variable_ops.ResourceVariable(initial_value=3.0, name='X.' + self.id())

        def f():
            if False:
                i = 10
                return i + 15
            return my_square(x)
        g = backprop.implicit_grad(f)
        grads_and_vars = g()
        self.assertEqual(1, len(grads_and_vars))
        (grad, var) = grads_and_vars[0]
        self.assertAllEqual(7, grad)
        self.assertAllEqual(x, var)

    def testJacobianCustomGradient(self):
        if False:
            i = 10
            return i + 15

        class MyCallable(object):

            def __init__(self):
                if False:
                    return 10
                self.a = variables.Variable(1.0)
                self.b = variables.Variable(2.0)
                self.c = variables.Variable(3.0)

            def __call__(self, x):
                if False:
                    print('Hello World!')
                return self.a * x * x + self.b * x + self.c

        @def_function.function
        def call(c, x):
            if False:
                print('Hello World!')

            @custom_gradient.custom_gradient
            def _call():
                if False:
                    for i in range(10):
                        print('nop')
                y = c(x)

                def grad(dy, variables=None):
                    if False:
                        while True:
                            i = 10
                    with backprop.GradientTape(persistent=True) as g:
                        g.watch(variables)
                        y = c(x)
                    grad_vars = [2 * math_ops.reduce_sum(dy * g.jacobian(y, v)) for v in variables]
                    del g
                    return ((), grad_vars)
                return (y, grad)
            return _call()
        c = MyCallable()
        x = constant_op.constant([1.0, 2.0, 3.0])
        with backprop.GradientTape(persistent=True) as g:
            g.watch([c.a, c.b, c.c])
            y = call(c, x)
        self.assertAllEqual(g.gradient(y, x), None)

    @test_util.assert_no_new_tensors
    def testCustomGradient(self):
        if False:
            while True:
                i = 10

        @custom_gradient.custom_gradient
        def my_mul(x, y):
            if False:
                for i in range(10):
                    print('nop')
            result = x * y

            def grad(dr):
                if False:
                    while True:
                        i = 10
                return [dr * y, dr * x]
            return (result, grad)
        lr = 0.25
        x = resource_variable_ops.ResourceVariable(2.0, name='x')

        def loss(x):
            if False:
                for i in range(10):
                    print('nop')
            return my_mul(2.0, x.read_value())
        loss_grads_fn = backprop.implicit_val_and_grad(loss)
        losses = []
        for _ in range(5):
            (loss, grads_and_vars) = loss_grads_fn(x)
            losses.append(loss.numpy())
            for (grad, var) in grads_and_vars:
                var.assign_sub(lr * grad)
        self.assertAllEqual(losses, [4.0, 3.0, 2.0, 1.0, 0.0])

    @test_util.assert_no_new_tensors
    def testCustomGradientIdentity(self):
        if False:
            i = 10
            return i + 15

        @custom_gradient.custom_gradient
        def my_identity(x):
            if False:
                return 10

            def grad(dresult):
                if False:
                    while True:
                        i = 10
                return [2 * dresult]
            return (x, grad)
        self.assertAllEqual(backprop.gradients_function(my_identity)(1.0)[0], 2.0)

    def testDifferentiatingFunctionThatReturnsNone(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            result = x * y
        x = constant_op.constant(1)
        y = constant_op.constant(2)
        loss_grads_fn = backprop.implicit_val_and_grad(fn)
        with self.assertRaisesRegex(ValueError, 'Cannot differentiate a function that returns None; did you forget to return a value from fn?'):
            loss_grads_fn(x, y)
        val_and_grads_fn = backprop.val_and_grad_function(fn)
        with self.assertRaisesRegex(ValueError, 'Cannot differentiate a function that returns None; did you forget to return a value from fn?'):
            val_and_grads_fn(x, y)

    def testZerosCacheDoesntLeakAcrossGraphs(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default():

            def get_grad():
                if False:
                    return 10
                with ops.Graph().as_default(), self.cached_session():
                    t = constant_op.constant(1, dtype=dtypes.float32, shape=(10, 4))
                    x = constant_op.constant(2, dtype=dtypes.float32, shape=(10, 4))
                    with backprop.GradientTape() as tape:
                        tape.watch(x)
                        (x1, _) = array_ops.split(x, num_or_size_splits=2, axis=1)
                        y1 = x1 ** 2
                        y = array_ops.concat([y1, t], axis=1)
                    return self.evaluate(tape.gradient(y, x))
            grad1 = get_grad()
            grad2 = get_grad()
            self.assertAllEqual(grad1, grad2)

    @test_util.run_in_graph_and_eager_modes
    def testSelectivelyWatchVariables(self):
        if False:
            i = 10
            return i + 15
        x1 = resource_variable_ops.ResourceVariable(1.0)
        x2 = resource_variable_ops.ResourceVariable(1.0)
        with backprop.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x2)
            y = x1 ** 2
            z = x2 ** 3
        self.assertTupleEqual(tape.watched_variables(), (x2,))
        (dy, dz) = tape.gradient([y, z], [x1, x2])
        self.evaluate([x1.initializer, x2.initializer])
        self.assertIsNone(dy)
        self.assertEqual(self.evaluate(dz), 3.0)

    @test_util.run_in_graph_and_eager_modes
    def testDifferentiatingScalarCache(self):
        if False:
            while True:
                i = 10
        with backprop.GradientTape(persistent=False) as g:
            x1 = constant_op.constant(3.0)
            x2 = constant_op.constant(3.0)
            g.watch(x1)
            g.watch(x2)
            y = x1 + x2
        grad = g.gradient(target=y, sources=[x1])
        self.assertEqual(self.evaluate(grad), [1.0])

    def testVariablesAndConstantsProduceTheSameGradients(self):
        if False:
            i = 10
            return i + 15

        def get_grads(a, b):
            if False:
                while True:
                    i = 10
            with backprop.GradientTape() as tape:
                tape.watch([a, b])
                y = a ** 3
                z = b ** 2
            return tape.gradient([y, z], [a, b])
        gradients_constants = get_grads(constant_op.constant(2.0), constant_op.constant(2.0))
        gradients_variables = get_grads(resource_variable_ops.ResourceVariable(2.0), resource_variable_ops.ResourceVariable(2.0))
        self.assertAllEqual(gradients_constants, gradients_variables)

    def testUnknownShapes(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default():
            with backprop.GradientTape() as tape:
                a = array_ops.placeholder(dtype=dtypes.float32, shape=None)
                tape.watch(a)
                b = a ** 3
            db_da = tape.gradient(b, a)
            with self.cached_session() as sess:
                self.assertEqual((8.0, 12.0), sess.run((b, db_da), feed_dict={a: 2.0}))

    @test_util.run_in_graph_and_eager_modes
    def testCustomGradientInEagerAndGraph(self):
        if False:
            print('Hello World!')

        @custom_gradient.custom_gradient
        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            y = x * x

            def grad(dy):
                if False:
                    for i in range(10):
                        print('nop')
                return [4 * dy]
            return (y, grad)
        with backprop.GradientTape() as t:
            c = constant_op.constant(1.0)
            t.watch(c)
            g = f(c)
        self.assertAllEqual(self.evaluate(t.gradient(g, c)), 4.0)

    def testOverrideSecondOrderWithCustomGradient(self):
        if False:
            return 10

        @custom_gradient.custom_gradient
        def f(x):
            if False:
                return 10

            def first_order_grad(dz):
                if False:
                    print('Hello World!')

                @custom_gradient.custom_gradient
                def first_order_custom(unused_x):
                    if False:
                        return 10

                    def h(ddz):
                        if False:
                            for i in range(10):
                                print('nop')
                        return -2.1 * ddz
                    return (-1.1, h)
                return dz * first_order_custom(x)
            return (x + 10.0, first_order_grad)
        c = constant_op.constant(1.0)
        with backprop.GradientTape() as outer:
            outer.watch(c)
            with backprop.GradientTape() as inner:
                inner.watch(c)
                d = f(c) ** 4.0
            dd = inner.gradient(d, c)
            self.assertAllClose(4.0 * f(c) ** 3.0 * -1.1, dd)
        self.assertAllClose(3.0 * 4.0 * f(c) ** 2.0 * -1.1 * -1.1 + 4.0 * f(c) ** 3.0 * -2.1, outer.gradient(dd, c))

    @test_util.run_in_graph_and_eager_modes
    def testCustomGradientForwardprop(self):
        if False:
            print('Hello World!')

        @custom_gradient.custom_gradient
        def f(x):
            if False:
                print('Hello World!')
            z = 2.0 * tensor_util.constant_value(x)

            def g(dz):
                if False:
                    return 10

                @custom_gradient.custom_gradient
                def first_order(unused_x, unused_dz):
                    if False:
                        for i in range(10):
                            print('nop')

                    def second_order_and_transpose(unused_ddz):
                        if False:
                            return 10
                        return (2.2, 3.1)
                    return (2.1, second_order_and_transpose)
                return first_order(x, dz)
            return (z, g)
        with backprop.GradientTape(persistent=True) as t:
            with backprop.GradientTape() as tt:
                c = constant_op.constant(1.0)
                t.watch(c)
                tt.watch(c)
                output_grad = array_ops.ones([])
                t.watch(output_grad)
                output = f(c)
                self.assertAllClose(2.0, output)
            gc = tt.gradient(output, c, output_gradients=output_grad)
            self.assertAllClose(2.1, gc)
        ggc = t.gradient(gc, c)
        self.assertAllClose(2.2, ggc)
        transpose = t.gradient(gc, output_grad)
        self.assertAllClose(3.1, transpose)

    @test_util.run_in_graph_and_eager_modes
    def testWatchBadThing(self):
        if False:
            return 10
        g = backprop.GradientTape()
        with self.assertRaisesRegex(ValueError, 'ndarray'):
            g.watch(np.array(1.0))

    def testWatchComposite(self):
        if False:
            i = 10
            return i + 15
        'Test that tape.watch expands composites and watches component Tensors.'
        with backprop.GradientTape() as t:
            values = constant_op.constant([1.0, 2.0], dtypes.float32)
            s = sparse_tensor.SparseTensor(indices=[[0, 0], [1, 2]], values=values, dense_shape=[3, 4])
            t.watch(s)
            z = sparse_ops.sparse_reduce_sum_v2(s)
        result = t.gradient(z, values)
        self.assertAllEqual(result, [1.0, 1.0])

    def testWatchedVariablesAfterNonPersistentGradientCall(self):
        if False:
            print('Hello World!')
        with backprop.GradientTape(persistent=False) as tape:
            x = resource_variable_ops.ResourceVariable(1.0)
            tape.watch(x)
        tape.gradient(x, x)
        self.assertEqual((x,), tape.watched_variables())

    def testWatchedVariablesOnlyHasVariablesFromLastTape(self):
        if False:
            for i in range(10):
                print('nop')
        with backprop.GradientTape(persistent=False) as tape:
            x = resource_variable_ops.ResourceVariable(1.0)
            tape.watch(x)
        with backprop.GradientTape(persistent=False) as tape:
            z = resource_variable_ops.ResourceVariable(2.0)
            tape.watch(z)
        tape.gradient(z, z)
        self.assertEqual((z,), tape.watched_variables())

    def testWatchedVariablesRespectReset(self):
        if False:
            while True:
                i = 10
        with backprop.GradientTape(persistent=False) as tape:
            x = resource_variable_ops.ResourceVariable(1.0)
            tape.watch(x)
            self.assertEqual((x,), tape.watched_variables())
            tape.reset()
            z = resource_variable_ops.ResourceVariable(2.0)
            tape.watch(z)
            self.assertEqual((z,), tape.watched_variables())
        tape.gradient(z, z)
        self.assertEqual((z,), tape.watched_variables())

    def testNameScope(self):
        if False:
            while True:
                i = 10

        def fn(x):
            if False:
                print('Hello World!')
            with ops.name_scope('my_scope'):
                a = math_ops.cos(x)
                b = math_ops.cos(x)
                return math_ops.add(a, b)

        @def_function.function
        def grad_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return backprop.gradients_function(fn)(x)
        grad_ops = grad_fn.get_concrete_function(constant_op.constant(1.0)).graph.get_operations()
        num_sin_ops_found = 0
        for op in grad_ops:
            if op.type == 'Sin':
                num_sin_ops_found += 1
                self.assertIn('gradient_tape/my_scope/', op.name)
        self.assertEqual(num_sin_ops_found, 2)

    @test_util.assert_no_new_pyobjects_executing_eagerly
    def testRecomputeGradWithDifferentShape(self):
        if False:
            return 10
        if sys.version_info.major == 3 and sys.version_info.minor in (11, 12):
            self.skipTest('Not working in Python 3.11+')

        @custom_gradient.recompute_grad
        def outer(x):
            if False:
                print('Hello World!')
            return [x[0] + 1, x[1] + 1]
        x = [variables.Variable([1.0, 2.0], name='a'), variables.Variable(1.0, name='b')]
        with backprop.GradientTape():
            y = outer(x)
            self.assertAllEqual(y[0], [2.0, 3.0])
            self.assertAllEqual(y[1], 2.0)

        @custom_gradient.recompute_grad
        def outer_dict(x):
            if False:
                print('Hello World!')
            for key in x.keys():
                x[key] = x[key] + 1
            return x
        x = {x[0].ref(): x[0], x[1].ref(): x[1]}
        with backprop.GradientTape():
            y = outer_dict(x)
            y = list(y.values())
            self.assertAllEqual(y[0], [2.0, 3.0])
            self.assertAllEqual(y[1], 2.0)

    @parameterized.parameters([True, False])
    @test_util.assert_no_new_pyobjects_executing_eagerly
    def testRecomputeGradWithNestedFunctionAndWhileLoop(self, reduce_retracing):
        if False:
            return 10
        if sys.version_info.major == 3 and sys.version_info.minor in (11, 12):
            self.skipTest('Not working in Python 3.11+')

        @custom_gradient.recompute_grad
        @def_function.function(reduce_retracing=reduce_retracing)
        def outer(x):
            if False:
                while True:
                    i = 10

            @def_function.function(reduce_retracing=reduce_retracing)
            def middle(y):
                if False:
                    return 10

                @def_function.function(reduce_retracing=reduce_retracing)
                def inner(z):
                    if False:
                        while True:
                            i = 10
                    return z + 1
                i = constant_op.constant(0.0)
                c = lambda y, i: i < 10.0
                b = lambda y, i: (inner(y), i + 1.0)
                (y, i) = while_loop.while_loop(c, b, [y, i])
                return y
            return middle(x)
        with MemoryChecker() as memory_checker:
            for _ in range(5):
                x = variables.Variable(1.0, name='x')
                with backprop.GradientTape():
                    y = outer(x)
                    self.assertAllEqual(y, 11.0)
        memory_checker.report()
        memory_checker.assert_no_leak_if_all_possibly_except_one()

class JacobianTest(test.TestCase):

    def _jacobian(self, experimental_use_pfor):
        if False:
            print('Hello World!')
        persistent = context.executing_eagerly and (not experimental_use_pfor)
        with backprop.GradientTape(persistent=persistent) as g:
            x = constant_op.constant([1.0, 2.0])
            y = constant_op.constant([3.0, 4.0])
            g.watch(x)
            g.watch(y)
            z = x * x * y
        jacobian = g.jacobian(z, [x, y], experimental_use_pfor=experimental_use_pfor)
        answer = [array_ops.diag(2 * x * y), array_ops.diag(x * x)]
        return (jacobian, answer)

    @test_util.run_v1_only('b/120545219')
    def testPfor(self):
        if False:
            while True:
                i = 10
        (jacobian, answer) = self._jacobian(experimental_use_pfor=True)
        for (j, a) in zip(jacobian, answer):
            self.assertAllEqual(a, j)

    @test_util.run_v1_only('b/120545219')
    def testWhileLoop(self):
        if False:
            i = 10
            return i + 15
        (jacobian, answer) = self._jacobian(experimental_use_pfor=False)
        for (j, a) in zip(jacobian, answer):
            self.assertAllEqual(a, j)

    @test_util.run_v1_only('b/120545219')
    def testPforDefun(self):
        if False:
            while True:
                i = 10

        @def_function.function
        def _f():
            if False:
                i = 10
                return i + 15
            return self._jacobian(experimental_use_pfor=True)
        (jacobian, answer) = _f()
        for (j, a) in zip(jacobian, answer):
            self.assertAllEqual(a, j)

    @test_util.run_v1_only('b/120545219')
    def testWhileLoopDefun(self):
        if False:
            return 10

        @def_function.function
        def _f():
            if False:
                while True:
                    i = 10
            return self._jacobian(experimental_use_pfor=False)
        (jacobian, answer) = _f()
        for (j, a) in zip(jacobian, answer):
            self.assertAllEqual(a, j)

    @test_util.run_v1_only('b/120545219')
    def testPersistentTape(self):
        if False:
            return 10
        if not context.executing_eagerly():
            return
        with backprop.GradientTape() as g:
            x = constant_op.constant([1.0, 2.0])
            g.watch(x)
            y = x * x
        with self.assertRaisesRegex(RuntimeError, 'persistent'):
            g.jacobian(y, x, experimental_use_pfor=False)

    @test_util.run_v1_only('b/120545219')
    def test_parallel_iterations(self):
        if False:
            i = 10
            return i + 15
        with backprop.GradientTape(persistent=True) as g:
            x = constant_op.constant([[1.0, 2], [3, 4]])
            g.watch(x)
            y = math_ops.matmul(x, x)
        self.assertAllClose(g.jacobian(y, x, parallel_iterations=2), g.jacobian(y, x, parallel_iterations=3))

    @test_util.run_in_graph_and_eager_modes
    def test_nested_jacobian(self):
        if False:
            print('Hello World!')
        if context.executing_eagerly():
            self.skipTest('Conversion of function calls not implemented yet.')
        x = array_ops.ones((10, 2))
        with backprop.GradientTape(persistent=False) as g:
            g.watch(x)
            with backprop.GradientTape(persistent=False) as gg:
                gg.watch(x)
                y = math_ops.reduce_sum(math_ops.square(x))
            dy_x = gg.jacobian(y, x)
        dy_xx = g.batch_jacobian(dy_x, x)
        dy_xx_answer = [[[2.0, 0], [0, 2.0]]] * 10
        self.assertAllClose(dy_xx_answer, self.evaluate(dy_xx))

    def test_nested_batch_jacobian_foldl(self):
        if False:
            while True:
                i = 10

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
                return tape.batch_jacobian(primal_out, primal)
            return _grad_function

        def _func(x):
            if False:
                while True:
                    i = 10
            return array_ops.reshape(functional_ops.foldl_v2(lambda a, b: math_ops.cos(a + b), array_ops.transpose(x)), [1, 1])
        f = _func
        x = constant_op.constant([[1.0, 2.0]])
        for _ in range(2):
            (theoretical, numerical) = gradient_checker_v2.compute_gradient(f, [x])
            self.assertAllClose(theoretical, numerical, rtol=0.001)
            f = _grad(f)
            expected_flat = array_ops.reshape(numerical, [-1])
            self.assertAllClose(expected_flat, array_ops.reshape(f(x), [-1]), rtol=0.001)
            self.assertAllClose(expected_flat, array_ops.reshape(def_function.function(f)(x), [-1]), rtol=0.001)

    def test_grad_jacobian_conv(self):
        if False:
            print('Hello World!')

        def _inner(x):
            if False:
                print('Hello World!')
            kernel = array_ops.ones([3, 3, 1, 9])
            with backprop.GradientTape() as tape:
                tape.watch(x)
                y = nn_ops.conv2d(x, kernel, strides=(1, 1), padding='SAME', data_format='NHWC')
                reduced = math_ops.reduce_sum(y ** 2.0, axis=[2, 3])
            return math_ops.reduce_sum(tape.batch_jacobian(reduced, x))
        (theoretical, numerical) = gradient_checker_v2.compute_gradient(def_function.function(_inner), [array_ops.ones([10, 4, 4, 1])])
        self.assertAllClose(numerical, theoretical, rtol=0.1)

        @def_function.function
        def _outer():
            if False:
                i = 10
                return i + 15
            with backprop.GradientTape() as tape:
                x = array_ops.ones([10, 4, 4, 1])
                tape.watch(x)
                y = _inner(x)
            return tape.gradient(y, x)
        self.assertAllClose(array_ops.reshape(numerical, [-1]), array_ops.reshape(_outer(), [-1]), rtol=0.1)

    @test_util.run_in_graph_and_eager_modes
    def test_indexed_slices(self):
        if False:
            print('Hello World!')
        with backprop.GradientTape(persistent=True) as g:
            inp = random_ops.random_uniform([3, 2])
            g.watch(inp)
            output = nn.embedding_lookup(inp, [0, 2])
        self.assertAllClose(g.jacobian(output, inp, experimental_use_pfor=True), g.jacobian(output, inp, experimental_use_pfor=False))

    def test_foldl_partial_function(self):
        if False:
            print('Hello World!')
        x = array_ops.zeros([3])
        with backprop.GradientTape(persistent=True) as tape:
            tape.watch(x)
            result = def_function.function(functools.partial(functional_ops.foldl_v2, lambda a, b: a + b))(x)
        self.assertAllClose([1.0, 1.0, 1.0], tape.jacobian(result, x, experimental_use_pfor=True))
        self.assertAllClose([1.0, 1.0, 1.0], tape.jacobian(result, x, experimental_use_pfor=False))
        x = array_ops.zeros([3])
        with backprop.GradientTape() as tape:
            tape.watch(x)
            result = def_function.function(functools.partial(functional_ops.foldl_v2, lambda a, b: a + b))(x)
        self.assertAllClose([1.0, 1.0, 1.0], tape.jacobian(result, x, experimental_use_pfor=True))

    def test_foldl_pure_function(self):
        if False:
            print('Hello World!')

        @def_function.function
        def compute_jacobian(use_pfor):
            if False:
                i = 10
                return i + 15
            x = array_ops.zeros([3])
            with backprop.GradientTape(persistent=True) as tape:
                tape.watch(x)
                result = functools.partial(functional_ops.foldl_v2, lambda a, b: a + b)(x)
            return tape.jacobian(result, x, experimental_use_pfor=use_pfor)
        self.assertAllClose(compute_jacobian(use_pfor=True), compute_jacobian(use_pfor=False))

    def test_cond_func_grad_jacobian(self):
        if False:
            print('Hello World!')

        @def_function.function
        def f(x):
            if False:
                i = 10
                return i + 15
            y = tf_cond.cond(x > 0.0, lambda : x ** 3.0, lambda : x ** 2.0)
            return y
        with backprop.GradientTape(persistent=True) as tape:
            x = constant_op.constant(1.0)
            tape.watch(x)
            y = f(x)
            grad = tape.gradient(y, x)
        self.assertAllClose(3.0, grad)
        jacobian = tape.jacobian(grad, x, experimental_use_pfor=False)
        self.assertAllClose(6.0, jacobian)
        jacobian_pfor = tape.jacobian(grad, x, experimental_use_pfor=True)
        self.assertAllClose(6.0, jacobian_pfor)

    def test_empty_tensor_consistent_jacobian(self):
        if False:
            i = 10
            return i + 15
        variable = variables.Variable(1.0)
        inputs = (constant_op.constant(np.random.uniform(size=(0, 4))), constant_op.constant(np.random.uniform(size=(0, 3))))
        with backprop.GradientTape(persistent=True) as tape:
            outputs = variable * math_ops.cast(array_ops.concat(inputs, axis=-1), dtypes.float32)
        jacobians_pfor = tape.jacobian(outputs, variable, experimental_use_pfor=True)
        jacobians_loop = tape.jacobian(outputs, variable, experimental_use_pfor=False)
        self.assertAllClose(jacobians_pfor, jacobians_loop)

@test_util.run_all_in_graph_and_eager_modes
class BatchJacobianTest(test.TestCase, parameterized.TestCase):

    def _batch_jacobian(self, experimental_use_pfor):
        if False:
            i = 10
            return i + 15
        persistent = context.executing_eagerly and (not experimental_use_pfor)
        with backprop.GradientTape(persistent=persistent) as g:
            x = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
            y = constant_op.constant([[3.0, 4.0], [5.0, 6.0]])
            g.watch(x)
            z = x * x * y
        batch_jacobian = g.batch_jacobian(z, x, experimental_use_pfor=experimental_use_pfor)
        answer = array_ops_stack.stack([array_ops.diag(2 * x[0] * y[0]), array_ops.diag(2 * x[1] * y[1])])
        return (batch_jacobian, answer)

    def testPfor(self):
        if False:
            for i in range(10):
                print('nop')
        (batch_jacobian, answer) = self._batch_jacobian(experimental_use_pfor=True)
        self.assertAllEqual(answer, batch_jacobian)

    def testWhileLoop(self):
        if False:
            return 10
        (batch_jacobian, answer) = self._batch_jacobian(experimental_use_pfor=False)
        self.assertAllEqual(answer, batch_jacobian)

    def testPforDefun(self):
        if False:
            print('Hello World!')

        @def_function.function
        def _f():
            if False:
                i = 10
                return i + 15
            return self._batch_jacobian(experimental_use_pfor=True)
        (batch_jacobian, answer) = _f()
        self.assertAllEqual(answer, batch_jacobian)

    def testWhileLoopDefun(self):
        if False:
            return 10

        @def_function.function
        def _f():
            if False:
                while True:
                    i = 10
            return self._batch_jacobian(experimental_use_pfor=False)
        (batch_jacobian, answer) = _f()
        self.assertAllEqual(answer, batch_jacobian)

    def testPersistentTape(self):
        if False:
            return 10
        if not context.executing_eagerly():
            return
        with backprop.GradientTape() as g:
            x = constant_op.constant([[1.0, 2.0]])
            g.watch(x)
            y = x * x
        with self.assertRaisesRegex(RuntimeError, 'persistent'):
            g.batch_jacobian(y, x, experimental_use_pfor=False)

    def testBadShape(self):
        if False:
            return 10
        x = random_ops.random_uniform([2, 3])
        with backprop.GradientTape() as g:
            y = array_ops.concat([x, x], axis=0)
        with self.assertRaisesRegex(ValueError, 'Need first dimension'):
            g.batch_jacobian(y, x)

    def testBadInputRank(self):
        if False:
            print('Hello World!')
        x = random_ops.random_uniform([2])
        with backprop.GradientTape() as g:
            y = random_ops.random_uniform([2, 2])
        with self.assertRaisesRegex(ValueError, 'must have rank at least 2'):
            g.batch_jacobian(y, x)

    def testBadOutputRank(self):
        if False:
            i = 10
            return i + 15
        x = random_ops.random_uniform([2, 2])
        with backprop.GradientTape() as g:
            y = random_ops.random_uniform([2])
        with self.assertRaisesRegex(ValueError, 'must have rank at least 2'):
            g.batch_jacobian(y, x)

    def test_parallel_iterations(self):
        if False:
            while True:
                i = 10
        with backprop.GradientTape(persistent=True) as g:
            x = constant_op.constant([[1.0, 2], [3, 4]])
            g.watch(x)
            w = constant_op.constant([[1.0, 2, 3, 4], [5, 6, 7, 8]])
            y = math_ops.matmul(x, w)
        self.assertAllClose(g.batch_jacobian(y, x, parallel_iterations=2), g.batch_jacobian(y, x, parallel_iterations=3))

    @parameterized.parameters((True, True), (True, False), (False, True), (False, False))
    def test_degenerate_shape(self, use_function, use_pfor):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                while True:
                    i = 10
            with backprop.GradientTape(persistent=True) as tape:
                tape.watch(x)
                y = x ** 2
            return tape.batch_jacobian(y, x, experimental_use_pfor=use_pfor)
        if use_function:
            f = def_function.function(f)
        self.assertAllEqual([1, 0, 0], array_ops.shape(f(array_ops.zeros([1, 0]))))

    @parameterized.parameters((True,), False)
    def test_zeros_type_correct(self, use_pfor):
        if False:
            for i in range(10):
                print('nop')
        for dtype in [dtypes.float32, dtypes.float64]:

            @def_function.function
            def f(x):
                if False:
                    for i in range(10):
                        print('nop')
                del x
                return constant_op.constant([[1.0]], dtype=dtype)
            with backprop.GradientTape(persistent=True) as tape:
                x = constant_op.constant([[2.0]], dtype=dtype)
                tape.watch(x)
                y = f(x)
            jac = tape.batch_jacobian(y, x, experimental_use_pfor=use_pfor)
            self.assertEqual(dtype, jac.dtype)
            self.assertAllClose([[[0.0]]], jac)
            with backprop.GradientTape(persistent=True) as tape:
                x = constant_op.constant([[2.0]], dtype=dtype)
                tape.watch(x)
                y = f(x)
            jac = tape.batch_jacobian(y, x, unconnected_gradients='zero', experimental_use_pfor=use_pfor)
            self.assertEqual(dtype, jac.dtype)
            self.assertAllClose([[[0.0]]], jac)

    def test_strided_slice(self):
        if False:
            print('Hello World!')
        x = array_ops.ones([2, 4, 2])
        length = constant_op.constant([2, 3, 4, 4], dtype=dtypes.int64)
        with backprop.GradientTape() as tape:
            tape.watch(x)
            y = array_ops.repeat(x, [2], axis=1)
            y = y[:, :math_ops.reduce_max(length), :]
        tape.batch_jacobian(y, x)

class AggregateIndexedSlicesGradientsTest(test_util.TensorFlowTestCase):

    def _assert_indexed_slices_equal(self, left, right):
        if False:
            i = 10
            return i + 15
        self.assertAllEqual(self.evaluate(ops.convert_to_tensor(left)), self.evaluate(ops.convert_to_tensor(right)))

    def testNoGradients(self):
        if False:
            return 10
        self.assertIsNone(backprop_util.AggregateIndexedSlicesGradients([]))

    def testOneGradient(self):
        if False:
            while True:
                i = 10
        t = math_ops._as_indexed_slices(constant_op.constant([[1.0, 2.0], [0, 0], [3.0, 4.0]]))
        result = backprop_util.AggregateIndexedSlicesGradients([t])
        self._assert_indexed_slices_equal(t, result)

    def testMultipleGradients(self):
        if False:
            for i in range(10):
                print('nop')
        t0 = math_ops._as_indexed_slices(constant_op.constant([[1.0, 2.0], [0, 0], [3.0, 4.0]]))
        t1 = math_ops._as_indexed_slices(constant_op.constant([[0.0, 0.0], [5, 6], [7.0, 8.0]]))
        total = constant_op.constant([[1.0, 2.0], [5, 6], [10.0, 12.0]])
        result = backprop_util.AggregateIndexedSlicesGradients([t0, t1])
        self._assert_indexed_slices_equal(total, result)

    def testMultipleGradientsWithNones(self):
        if False:
            return 10
        t0 = math_ops._as_indexed_slices(constant_op.constant([[1.0, 2.0], [0, 0], [3.0, 4.0]]))
        t1 = math_ops._as_indexed_slices(constant_op.constant([[0.0, 0.0], [5, 6], [7.0, 8.0]]))
        t3 = None
        total = constant_op.constant([[1.0, 2.0], [5, 6], [10.0, 12.0]])
        result = backprop_util.AggregateIndexedSlicesGradients([t0, t1, t3])
        self._assert_indexed_slices_equal(total, result)

    def testMixedTensorAndIndexedSlices(self):
        if False:
            for i in range(10):
                print('nop')
        t0 = math_ops._as_indexed_slices(constant_op.constant([[1.0, 2.0], [0, 0], [3.0, 4.0]]))
        t1 = constant_op.constant([[0.0, 0.0], [5, 6], [7.0, 8.0]])
        total = constant_op.constant([[1.0, 2.0], [5, 6], [10.0, 12.0]])
        result = backprop_util.AggregateIndexedSlicesGradients([t0, t1])
        self._assert_indexed_slices_equal(total, result)
if __name__ == '__main__':
    test.main()