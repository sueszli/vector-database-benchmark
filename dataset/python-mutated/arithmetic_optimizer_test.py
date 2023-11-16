"""Tests for Grappler Arithmetic Optimizer."""
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class ArithmeticOptimizerTest(test.TestCase):

    def testFunctionArgShapeInference(self):
        if False:
            for i in range(10):
                print('nop')

        @def_function.function
        def f(x, y):
            if False:
                return 10
            return math_ops.matmul(x, array_ops.reshape(array_ops.transpose(y), [384, 1536]))
        with context.eager_mode():
            x = array_ops.ones((1, 384))
            y = array_ops.ones((1536, 384))
            with context.collect_graphs(optimized=True) as graphs:
                f(x, y).numpy()
            self.assertLen(graphs, 1)
            self.assertLen(graphs[0].node, 4)
            self.assertEqual(graphs[0].node[2].name, 'ArithmeticOptimizer/FoldTransposeIntoMatMul_MatMul')
if __name__ == '__main__':
    test.main()