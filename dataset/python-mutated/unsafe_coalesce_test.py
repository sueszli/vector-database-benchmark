import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
from caffe2.python import core, workspace
from hypothesis import given

class TestUnsafeCoalesceOp(hu.HypothesisTestCase):

    @given(n=st.integers(1, 5), shape=st.lists(st.integers(0, 5), min_size=1, max_size=3), **hu.gcs)
    def test_unsafe_coalesce_op(self, n, shape, dc, gc):
        if False:
            print('Hello World!')
        workspace.ResetWorkspace()
        test_inputs = [(100 * np.random.random(shape)).astype(np.float32) for _ in range(n)]
        test_input_blobs = ['x_{}'.format(i) for i in range(n)]
        coalesce_op = core.CreateOperator('UnsafeCoalesce', test_input_blobs, test_input_blobs + ['shared_memory_blob'], device_option=gc)

        def reference_func(*args):
            if False:
                return 10
            self.assertEqual(len(args), n)
            return list(args) + [np.concatenate([x.flatten() for x in args])]
        self.assertReferenceChecks(gc, coalesce_op, test_inputs, reference_func)

    @given(n=st.integers(1, 5), shape=st.lists(st.integers(1, 5), min_size=1, max_size=3), seed=st.integers(0, 65535), **hu.gcs)
    def test_unsafe_coalesce_op_blob_sharing(self, n, shape, seed, dc, gc):
        if False:
            i = 10
            return i + 15
        workspace.ResetWorkspace()
        np.random.seed(seed)
        test_inputs = [np.random.random(shape).astype(np.float32) for _ in range(n)]
        test_input_blobs = ['x_{}'.format(i) for i in range(n)]
        coalesce_op = core.CreateOperator('UnsafeCoalesce', test_input_blobs, test_input_blobs + ['shared_memory_blob'], device_option=gc)
        for (name, value) in zip(test_input_blobs, test_inputs):
            workspace.FeedBlob(name, value, device_option=gc)
        workspace.RunOperatorOnce(coalesce_op)
        blob_value = workspace.blobs['shared_memory_blob']
        npt.assert_almost_equal(blob_value, np.concatenate([x.flatten() for x in test_inputs]), decimal=4)
        blob_value.fill(-2.0)
        self.assertTrue((blob_value != workspace.blobs['shared_memory_blob']).all())
        workspace.FeedBlob('shared_memory_blob', blob_value, device_option=gc)
        for (name, value) in zip(test_input_blobs, test_inputs):
            self.assertEqual(value.shape, workspace.blobs[name].shape)
            self.assertTrue((value != workspace.blobs[name]).all())
            self.assertTrue((workspace.blobs[name] == -2).all())
        workspace.RunOperatorOnce(coalesce_op)