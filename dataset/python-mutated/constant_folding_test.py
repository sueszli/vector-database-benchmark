"""Tests for Grappler Constant Folding."""
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test

class ConstantFoldingTest(test.TestCase):

    def testScanInsideWhile(self):
        if False:
            return 10

        def loop_cond(idx_step, *unused_args):
            if False:
                i = 10
                return i + 15
            return idx_step < 1

        def loop_body(idx_step, y):
            if False:
                for i in range(10):
                    print('nop')
            x = array_ops.zeros([10, 20, 30], dtype=dtypes.float32)
            x = functional_ops.scan(math_ops.add, x, initializer=array_ops.zeros([20, 30], dtype=dtypes.float32), back_prop=False, parallel_iterations=1)
            with ops.device('/cpu:0'):
                y = array_ops.identity(x)
                return (idx_step + 1, y)
        if test.is_gpu_available(cuda_only=True):
            init_y = array_ops.zeros([10, 20, 30], dtype=dtypes.float32)
            (_, y) = while_loop.while_loop(loop_cond, loop_body, loop_vars=[0, init_y], back_prop=False, parallel_iterations=1)
            y_v = self.evaluate(y)
            self.assertAllEqual(np.zeros([10, 20, 30]), y_v)

    def testGradientGraphOptimization(self):
        if False:
            while True:
                i = 10

        @def_function.function
        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            with backprop.GradientTape() as tape:
                z = math_ops.mul(x, array_ops.zeros_like(x))
                l = math_ops.add(z, y)
                l = math_ops.reduce_sum(l)
            (gx, gy) = tape.gradient(l, [x, y])
            x.assign_add(gx)
            y.assign_add(gy)
            return x + y
        if test_util.is_xla_enabled():
            self.skipTest('Not relevant for XLA')
        with context.eager_mode():
            x = resource_variable_ops.ResourceVariable(np.random.uniform(size=[2, 2]), dtype=dtypes.float32)
            y = resource_variable_ops.ResourceVariable(np.random.uniform(size=[2, 2]), dtype=dtypes.float32)
            with context.collect_graphs(optimized=True) as graphs:
                f(x, y).numpy()
        self.assertLen(graphs, 1)
        assign_count = 0
        for node in graphs[0].node:
            if node.op == 'AssignAddVariableOp':
                self.assertEqual(node.input[0], 'y')
                assign_count += 1
        self.assertEqual(assign_count, 1)
        self.assertLen(graphs[0].node, 11)
if __name__ == '__main__':
    test.main()