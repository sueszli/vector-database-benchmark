from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import given, settings
import hypothesis.strategies as st
import numpy as np

class TestMarginRankingCriterion(serial.SerializedTestCase):

    @given(N=st.integers(min_value=10, max_value=20), seed=st.integers(min_value=0, max_value=65535), margin=st.floats(min_value=-0.5, max_value=0.5), **hu.gcs)
    @settings(deadline=10000)
    def test_margin_ranking_criterion(self, N, seed, margin, gc, dc):
        if False:
            for i in range(10):
                print('nop')
        np.random.seed(seed)
        X1 = np.random.randn(N).astype(np.float32)
        X2 = np.random.randn(N).astype(np.float32)
        Y = np.random.choice([-1, 1], size=N).astype(np.int32)
        op = core.CreateOperator('MarginRankingCriterion', ['X1', 'X2', 'Y'], ['loss'], margin=margin)

        def ref_cec(X1, X2, Y):
            if False:
                print('Hello World!')
            result = np.maximum(-Y * (X1 - X2) + margin, 0)
            return (result,)
        inputs = [X1, X2, Y]
        self.assertReferenceChecks(gc, op, inputs, ref_cec)
        self.assertDeviceChecks(dc, op, inputs, [0])
        X1[np.abs(margin - Y * (X1 - X2)) < 0.1] += 0.1
        X2[np.abs(margin - Y * (X1 - X2)) < 0.1] -= 0.1
        self.assertGradientChecks(gc, op, inputs, 0, [0])
        self.assertGradientChecks(gc, op, inputs, 1, [0])
if __name__ == '__main__':
    import unittest
    unittest.main()