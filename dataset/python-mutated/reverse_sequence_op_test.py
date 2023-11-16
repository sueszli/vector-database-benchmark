"""Tests for tensorflow.ops.reverse_sequence_op."""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class ReverseSequenceTest(xla_test.XLATestCase):

    def _testReverseSequence(self, x, batch_axis, seq_axis, seq_lengths, truth, expected_err_re=None):
        if False:
            while True:
                i = 10
        with self.session():
            p = array_ops.placeholder(dtypes.as_dtype(x.dtype))
            lengths = array_ops.placeholder(dtypes.as_dtype(seq_lengths.dtype))
            with self.test_scope():
                ans = array_ops.reverse_sequence(p, batch_axis=batch_axis, seq_axis=seq_axis, seq_lengths=lengths)
            if expected_err_re is None:
                tf_ans = ans.eval(feed_dict={p: x, lengths: seq_lengths})
                self.assertAllClose(tf_ans, truth, atol=1e-10)
            else:
                with self.assertRaisesOpError(expected_err_re):
                    ans.eval(feed_dict={p: x, lengths: seq_lengths})

    def testSimple(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
        expected = np.array([[1, 2, 3], [6, 5, 4], [8, 7, 9]], dtype=np.int32)
        self._testReverseSequence(x, batch_axis=0, seq_axis=1, seq_lengths=np.array([1, 3, 2], np.int32), truth=expected)

    def _testBasic(self, dtype, len_dtype):
        if False:
            i = 10
            return i + 15
        x = np.asarray([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]], [[17, 18, 19, 20], [21, 22, 23, 24]]], dtype=dtype)
        x = x.reshape(3, 2, 4, 1, 1)
        x = x.transpose([2, 1, 0, 3, 4])
        seq_lengths = np.asarray([3, 0, 4], dtype=len_dtype)
        truth_orig = np.asarray([[[3, 2, 1, 4], [7, 6, 5, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]], [[20, 19, 18, 17], [24, 23, 22, 21]]], dtype=dtype)
        truth_orig = truth_orig.reshape(3, 2, 4, 1, 1)
        truth = truth_orig.transpose([2, 1, 0, 3, 4])
        seq_axis = 0
        batch_axis = 2
        self._testReverseSequence(x, batch_axis, seq_axis, seq_lengths, truth)

    def testSeqLength(self):
        if False:
            print('Hello World!')
        for dtype in self.all_types:
            for seq_dtype in self.all_types & {np.int32, np.int64}:
                self._testBasic(dtype, seq_dtype)
if __name__ == '__main__':
    test.main()