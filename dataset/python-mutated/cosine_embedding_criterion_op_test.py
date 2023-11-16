import hypothesis.strategies as st
import numpy as np
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

class TestCosineEmbeddingCriterion(serial.SerializedTestCase):

    @serial.given(N=st.integers(min_value=10, max_value=20), seed=st.integers(min_value=0, max_value=65535), margin=st.floats(min_value=-0.5, max_value=0.5), **hu.gcs)
    def test_cosine_embedding_criterion(self, N, seed, margin, gc, dc):
        if False:
            print('Hello World!')
        np.random.seed(seed)
        S = np.random.randn(N).astype(np.float32)
        Y = np.random.choice([-1, 1], size=N).astype(np.int32)
        op = core.CreateOperator('CosineEmbeddingCriterion', ['S', 'Y'], ['output'], margin=margin)

        def ref_cec(S, Y):
            if False:
                for i in range(10):
                    print('nop')
            result = (1 - S) * (Y == 1) + np.maximum(S - margin, 0) * (Y == -1)
            return (result,)
        self.assertReferenceChecks(gc, op, [S, Y], ref_cec)
        self.assertDeviceChecks(dc, op, [S, Y], [0])
        S[np.abs(S - margin) < 0.1] += 0.2
        self.assertGradientChecks(gc, op, [S, Y], 0, [0])
if __name__ == '__main__':
    import unittest
    unittest.main()