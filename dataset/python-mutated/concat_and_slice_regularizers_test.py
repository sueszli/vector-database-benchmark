"""Tests for framework.concat_and_slice_regularizers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from morph_net.framework import concat_and_slice_regularizers
from morph_net.testing import op_regularizer_stub

class ConcatAndSliceRegularizersTest(tf.test.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._reg_vec1 = [0.1, 0.3, 0.6, 0.2]
        self._alive_vec1 = [False, True, True, False]
        self._reg_vec2 = [0.2, 0.4, 0.5]
        self._alive_vec2 = [False, True, False]
        self._reg1 = op_regularizer_stub.OpRegularizerStub(self._reg_vec1, self._alive_vec1)
        self._reg2 = op_regularizer_stub.OpRegularizerStub(self._reg_vec2, self._alive_vec2)

    def testConcatRegularizer(self):
        if False:
            return 10
        concat_reg = concat_and_slice_regularizers.ConcatRegularizer([self._reg1, self._reg2])
        with self.test_session():
            self.assertAllEqual(self._alive_vec1 + self._alive_vec2, concat_reg.alive_vector.eval())
            self.assertAllClose(self._reg_vec1 + self._reg_vec2, concat_reg.regularization_vector.eval(), 1e-05)

    def testSliceRegularizer(self):
        if False:
            while True:
                i = 10
        concat_reg = concat_and_slice_regularizers.SlicingReferenceRegularizer(lambda : self._reg1, 1, 2)
        with self.test_session():
            self.assertAllEqual(self._alive_vec1[1:3], concat_reg.alive_vector.eval())
            self.assertAllClose(self._reg_vec1[1:3], concat_reg.regularization_vector.eval(), 1e-05)
if __name__ == '__main__':
    tf.test.main()