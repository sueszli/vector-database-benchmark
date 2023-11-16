from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np

class TestATen(hu.HypothesisTestCase):

    @given(inputs=hu.tensors(n=2), **hu.gcs)
    def test_add(self, inputs, gc, dc):
        if False:
            for i in range(10):
                print('nop')
        op = core.CreateOperator('ATen', ['X', 'Y'], ['Z'], operator='add')

        def ref(X, Y):
            if False:
                print('Hello World!')
            return [X + Y]
        self.assertReferenceChecks(gc, op, inputs, ref)

    @given(inputs=hu.tensors(n=2, dtype=np.float16), **hu.gcs_gpu_only)
    def test_add_half(self, inputs, gc, dc):
        if False:
            for i in range(10):
                print('nop')
        op = core.CreateOperator('ATen', ['X', 'Y'], ['Z'], operator='add')

        def ref(X, Y):
            if False:
                print('Hello World!')
            return [X + Y]
        self.assertReferenceChecks(gc, op, inputs, ref)

    @given(inputs=hu.tensors(n=1), **hu.gcs)
    def test_pow(self, inputs, gc, dc):
        if False:
            i = 10
            return i + 15
        op = core.CreateOperator('ATen', ['S'], ['Z'], operator='pow', exponent=2.0)

        def ref(X):
            if False:
                return 10
            return [np.square(X)]
        self.assertReferenceChecks(gc, op, inputs, ref)

    @given(x=st.integers(min_value=2, max_value=8), **hu.gcs)
    def test_sort(self, x, gc, dc):
        if False:
            while True:
                i = 10
        inputs = [np.random.permutation(x)]
        op = core.CreateOperator('ATen', ['S'], ['Z', 'I'], operator='sort')

        def ref(X):
            if False:
                for i in range(10):
                    print('nop')
            return [np.sort(X), np.argsort(X)]
        self.assertReferenceChecks(gc, op, inputs, ref)

    @given(inputs=hu.tensors(n=1), **hu.gcs)
    def test_sum(self, inputs, gc, dc):
        if False:
            while True:
                i = 10
        op = core.CreateOperator('ATen', ['S'], ['Z'], operator='sum')

        def ref(X):
            if False:
                print('Hello World!')
            return [np.sum(X)]
        self.assertReferenceChecks(gc, op, inputs, ref)

    @given(**hu.gcs)
    def test_index_uint8(self, gc, dc):
        if False:
            print('Hello World!')
        op = core.CreateOperator('ATen', ['self', 'mask'], ['Z'], operator='index')

        def ref(self, mask):
            if False:
                for i in range(10):
                    print('nop')
            return (self[mask.astype(np.bool_)],)
        tensor = np.random.randn(2, 3, 4).astype(np.float32)
        mask = np.array([[1, 0, 0], [1, 1, 0]]).astype(np.uint8)
        self.assertReferenceChecks(gc, op, [tensor, mask], ref)

    @given(**hu.gcs)
    def test_index_put(self, gc, dc):
        if False:
            i = 10
            return i + 15
        op = core.CreateOperator('ATen', ['self', 'indices', 'values'], ['Z'], operator='index_put')

        def ref(self, indices, values):
            if False:
                while True:
                    i = 10
            self[indices] = values
            return (self,)
        tensor = np.random.randn(3, 3).astype(np.float32)
        mask = np.array([[True, True, True], [True, False, False], [True, True, False]])
        values = np.random.randn(6).astype(np.float32)
        self.assertReferenceChecks(gc, op, [tensor, mask, values], ref)

    @given(**hu.gcs)
    def test_unique(self, gc, dc):
        if False:
            while True:
                i = 10
        op = core.CreateOperator('ATen', ['self'], ['output'], sorted=True, return_inverse=True, operator='_unique')

        def ref(self):
            if False:
                for i in range(10):
                    print('nop')
            (index, _) = np.unique(self, return_index=False, return_inverse=True, return_counts=False)
            return (index,)
        tensor = np.array([1, 2, 6, 4, 2, 3, 2])
        print(ref(tensor))
        self.assertReferenceChecks(gc, op, [tensor], ref)
if __name__ == '__main__':
    import unittest
    unittest.main()