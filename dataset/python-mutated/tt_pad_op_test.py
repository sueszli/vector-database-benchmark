import numpy as np
from hypothesis import given
import hypothesis.strategies as st
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu

class TestTTPad(hu.HypothesisTestCase):

    @given(K=st.integers(min_value=2, max_value=10), M=st.integers(min_value=10, max_value=20), N=st.integers(min_value=10, max_value=20), **hu.gcs)
    def test_tt_pad(self, K, M, N, gc, dc):
        if False:
            for i in range(10):
                print('nop')
        op = core.CreateOperator('TTPad', ['A'], ['A', 'dim0'], scale=K)
        A = np.random.rand(M, N).astype(np.float32)
        workspace.FeedBlob('A', A)
        workspace.RunOperatorOnce(op)

        def tt_pad_ref(A_):
            if False:
                i = 10
                return i + 15
            M_ = A_.shape[0]
            if M_ % K == 0:
                new_dim0 = M_
            else:
                new_dim0 = (M_ // K + 1) * K
            return (np.vstack((A_, np.zeros((new_dim0 - M_, A_.shape[1])))), np.array([A.shape[0]]))
        self.assertReferenceChecks(gc, op, [A], tt_pad_ref)
        self.assertDeviceChecks(dc, op, [A], [0])
        self.assertGradientChecks(gc, op, [A], 0, [0])