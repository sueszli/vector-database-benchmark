"""Functional test for GradientDescent."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent

class GradientDescentOptimizerTest(test.TestCase):

    def testBasic(self):
        if False:
            print('Hello World!')
        for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
            with ops.Graph().as_default(), self.cached_session():
                var0 = variables.Variable([1.0, 2.0], dtype=dtype)
                var1 = variables.Variable([3.0, 4.0], dtype=dtype)
                grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
                grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
                optimizer = gradient_descent.GradientDescentOptimizer(3.0)
                sgd_op = optimizer.apply_gradients(zip([grads0, grads1], [var0, var1]))
                self.evaluate(variables.global_variables_initializer())
                self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
                self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
                sgd_op.run()
                self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1], self.evaluate(var0))
                self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01], self.evaluate(var1))
                self.assertEqual(0, len(optimizer.variables()))

    def testBasicResourceVariable(self):
        if False:
            print('Hello World!')
        for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
            with ops.Graph().as_default(), self.cached_session():
                var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
                var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
                grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
                grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
                sgd_op = gradient_descent.GradientDescentOptimizer(3.0).apply_gradients(zip([grads0, grads1], [var0, var1]))
                resources.initialize_resources([var0, var1]).run()
                self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
                self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
                sgd_op.run()
                self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1], self.evaluate(var0))
                self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01], self.evaluate(var1))

    def testBasicCallableParams(self):
        if False:
            print('Hello World!')
        for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
            with ops.Graph().as_default(), self.cached_session():
                var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
                var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
                grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
                grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
                lr = lambda : 3.0
                sgd_op = gradient_descent.GradientDescentOptimizer(lr).apply_gradients(zip([grads0, grads1], [var0, var1]))
                resources.initialize_resources([var0, var1]).run()
                self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
                self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
                sgd_op.run()
                self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1], self.evaluate(var0))
                self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01], self.evaluate(var1))

    def testMinimizeResourceVariable(self):
        if False:
            return 10
        for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
            with ops.Graph().as_default(), self.cached_session():
                var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
                var1 = resource_variable_ops.ResourceVariable([3.0], dtype=dtype)
                x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
                pred = math_ops.matmul(var0, x) + var1
                loss = pred * pred
                sgd_op = gradient_descent.GradientDescentOptimizer(1.0).minimize(loss)
                resources.initialize_resources([var0, var1]).run()
                self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
                self.assertAllCloseAccordingToType([3.0], self.evaluate(var1))
                sgd_op.run()
                np_pred = 1.0 * 4.0 + 2.0 * 5.0 + 3.0
                np_grad = 2 * np_pred
                self.assertAllCloseAccordingToType([[1.0 - np_grad * 4.0, 2.0 - np_grad * 5.0]], self.evaluate(var0))
                self.assertAllCloseAccordingToType([3.0 - np_grad], self.evaluate(var1))

    def testMinimizeSparseResourceVariable(self):
        if False:
            i = 10
            return i + 15
        for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
            with ops.Graph().as_default(), self.cached_session():
                var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
                var1 = resource_variable_ops.ResourceVariable([3.0], dtype=dtype)
                x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
                pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
                pred += var1
                loss = pred * pred
                sgd_op = gradient_descent.GradientDescentOptimizer(1.0).minimize(loss)
                self.evaluate(variables.global_variables_initializer())
                self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
                self.assertAllCloseAccordingToType([3.0], self.evaluate(var1))
                sgd_op.run()
                np_pred = 1.0 * 4.0 + 2.0 * 5.0 + 3.0
                np_grad = 2 * np_pred
                self.assertAllCloseAccordingToType([[1.0 - np_grad * 4.0, 2.0 - np_grad * 5.0]], self.evaluate(var0))
                self.assertAllCloseAccordingToType([3.0 - np_grad], self.evaluate(var1))

    def testTensorLearningRate(self):
        if False:
            for i in range(10):
                print('nop')
        for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
            with ops.Graph().as_default(), self.cached_session():
                var0 = variables.Variable([1.0, 2.0], dtype=dtype)
                var1 = variables.Variable([3.0, 4.0], dtype=dtype)
                grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
                grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
                lrate = constant_op.constant(3.0)
                sgd_op = gradient_descent.GradientDescentOptimizer(lrate).apply_gradients(zip([grads0, grads1], [var0, var1]))
                self.evaluate(variables.global_variables_initializer())
                self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
                self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
                sgd_op.run()
                self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1], self.evaluate(var0))
                self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01], self.evaluate(var1))

    def testGradWrtRef(self):
        if False:
            return 10
        for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
            with ops.Graph().as_default(), self.cached_session():
                opt = gradient_descent.GradientDescentOptimizer(3.0)
                values = [1.0, 3.0]
                vars_ = [variables.Variable([v], dtype=dtype) for v in values]
                grads_and_vars = opt.compute_gradients(vars_[0] + vars_[1], vars_)
                self.evaluate(variables.global_variables_initializer())
                for (grad, _) in grads_and_vars:
                    self.assertAllCloseAccordingToType([1.0], self.evaluate(grad))

    def testWithGlobalStep(self):
        if False:
            i = 10
            return i + 15
        for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
            with ops.Graph().as_default(), self.cached_session():
                global_step = variables.Variable(0, trainable=False)
                var0 = variables.Variable([1.0, 2.0], dtype=dtype)
                var1 = variables.Variable([3.0, 4.0], dtype=dtype)
                grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
                grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
                sgd_op = gradient_descent.GradientDescentOptimizer(3.0).apply_gradients(zip([grads0, grads1], [var0, var1]), global_step=global_step)
                self.evaluate(variables.global_variables_initializer())
                self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
                self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
                sgd_op.run()
                self.assertAllCloseAccordingToType([1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1], self.evaluate(var0))
                self.assertAllCloseAccordingToType([3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01], self.evaluate(var1))
                self.assertAllCloseAccordingToType(1, self.evaluate(global_step))

    def testSparseBasic(self):
        if False:
            i = 10
            return i + 15
        for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
            with ops.Graph().as_default(), self.cached_session():
                var0 = variables.Variable([[1.0], [2.0]], dtype=dtype)
                var1 = variables.Variable([[3.0], [4.0]], dtype=dtype)
                grads0 = indexed_slices.IndexedSlices(constant_op.constant([0.1], shape=[1, 1], dtype=dtype), constant_op.constant([0]), constant_op.constant([2, 1]))
                grads1 = indexed_slices.IndexedSlices(constant_op.constant([0.01], shape=[1, 1], dtype=dtype), constant_op.constant([1]), constant_op.constant([2, 1]))
                sgd_op = gradient_descent.GradientDescentOptimizer(3.0).apply_gradients(zip([grads0, grads1], [var0, var1]))
                self.evaluate(variables.global_variables_initializer())
                self.assertAllCloseAccordingToType([[1.0], [2.0]], self.evaluate(var0))
                self.assertAllCloseAccordingToType([[3.0], [4.0]], self.evaluate(var1))
                sgd_op.run()
                self.assertAllCloseAccordingToType([[1.0 - 3.0 * 0.1], [2.0]], self.evaluate(var0))
                self.assertAllCloseAccordingToType([[3.0], [4.0 - 3.0 * 0.01]], self.evaluate(var1))
if __name__ == '__main__':
    test.main()