"""Tests for array_grad."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.platform import test

@test_util.with_eager_op_as_function
@test_util.run_all_in_graph_and_eager_modes
class ArrayGradTest(test.TestCase):

    def _testGrad(self, f, x):
        if False:
            return 10
        max_error = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(f, [x]))
        self.assertLess(max_error, 0.0001)

    def test_gather_v2_simple(self):
        if False:
            return 10
        x = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtypes.float64)

        def f(x):
            if False:
                print('Hello World!')
            return array_ops.gather_v2(x, constant_op.constant([2, 0, 2, 4], dtype=dtypes.int32))
        self._testGrad(f, x)

    def test_gather_v2_more_index_dims(self):
        if False:
            while True:
                i = 10
        x = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtypes.float64)

        def f(x):
            if False:
                print('Hello World!')
            return array_ops.gather_v2(x, constant_op.constant([[2, 0], [2, 4]], dtype=dtypes.int32))
        self._testGrad(f, x)

    def test_gather_v2_more_param_dims(self):
        if False:
            for i in range(10):
                print('nop')
        x = constant_op.constant([[1.0, 2.0], [3.0, 4.0]], dtype=dtypes.float64)

        def f(x):
            if False:
                i = 10
                return i + 15
            return array_ops.gather_v2(x, constant_op.constant([1, 0], dtype=dtypes.int32))
        self._testGrad(f, x)

    def test_gather_v2_axis(self):
        if False:
            i = 10
            return i + 15
        x = constant_op.constant([[1.0, 2.0], [3.0, 4.0]], dtype=dtypes.float64)

        def f(x):
            if False:
                i = 10
                return i + 15
            return array_ops.gather_v2(x, constant_op.constant([1, 0], dtype=dtypes.int32), axis=1)
        self._testGrad(f, x)

    def test_gather_v2_batch_dims(self):
        if False:
            for i in range(10):
                print('nop')
        x = constant_op.constant([[1.0, 2.0], [3.0, 4.0]], dtype=dtypes.float64)

        def f(x):
            if False:
                return 10
            return array_ops.gather_v2(x, constant_op.constant([[1, 0], [0, 0]], dtype=dtypes.int32), axis=1, batch_dims=1)
        self._testGrad(f, x)

    def test_gather_v2_2batch_dims(self):
        if False:
            return 10
        x = constant_op.constant([[[1.0, 2.0], [3.0, 4.0]]], dtype=dtypes.float64)

        def f(x):
            if False:
                while True:
                    i = 10
            return array_ops.gather_v2(x, constant_op.constant([[[1, 0], [0, 0]]], dtype=dtypes.int32), axis=2, batch_dims=2)
        self._testGrad(f, x)

    def test_gather_v2_batch_dims_with_axis(self):
        if False:
            print('Hello World!')
        x = constant_op.constant([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=dtypes.float64)

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return array_ops.gather_v2(x, constant_op.constant([[0], [0]], dtype=dtypes.int32), axis=2, batch_dims=1)
        self._testGrad(f, x)

    def test_broadcast_to(self):
        if False:
            return 10
        x = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float64)
        y = constant_op.constant([2, 3], dtype=dtypes.int32)

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return array_ops.broadcast_to(x, y)
        self._testGrad(f, x)

    def test_broadcast_to_int64(self):
        if False:
            print('Hello World!')
        x = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float64)
        y = constant_op.constant([2, 3], dtype=dtypes.int64)

        def f(x):
            if False:
                print('Hello World!')
            return array_ops.broadcast_to(x, y)
        self._testGrad(f, x)

    def test_slice_int64(self):
        if False:
            while True:
                i = 10
        x = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float64)
        begin = constant_op.constant([1], dtype=dtypes.int64)
        size = constant_op.constant([1], dtype=dtypes.int64)

        def f(x):
            if False:
                i = 10
                return i + 15
            return array_ops.slice(x, begin, size)
        self._testGrad(f, x)
if __name__ == '__main__':
    test.main()