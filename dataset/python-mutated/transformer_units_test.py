"""Tests for dragnn.python.transformer_units."""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from dragnn.python import transformer_units

class TransformerTest(test_util.TensorFlowTestCase):

    def testComputePadding(self):
        if False:
            return 10
        with tf.Graph().as_default(), self.test_session() as session:
            lengths = [5, 1, 2, 0]
            expected = [[[[0, 0, 0, 0, 0]]], [[[0, -1000000000.0, -1000000000.0, -1000000000.0, -1000000000.0]]], [[[0, 0, -1000000000.0, -1000000000.0, -1000000000.0]]], [[[-1000000000.0, -1000000000.0, -1000000000.0, -1000000000.0, -1000000000.0]]]]
            tensor = transformer_units.compute_padding_mask(lengths)
            session.run(tf.global_variables_initializer())
            actual = session.run(tensor)
            self.assertAllEqual(actual, expected)

    def testDotProductAttention(self):
        if False:
            while True:
                i = 10
        with tf.Graph().as_default(), self.test_session() as session:
            padding = [[[[0, 0, 0, 0, 0]]], [[[0, -1000000000.0, -1000000000.0, -1000000000.0, -1000000000.0]]]]
            np.random.seed(4)
            q = np.random.random((2, 2, 5, 2)).astype(np.float32)
            k = np.random.random((2, 2, 5, 2)).astype(np.float32)
            v = np.random.random((2, 2, 5, 2)).astype(np.float32)
            expected = [[[[0.46580601, 0.64643575], [0.46182397, 0.64578158], [0.46866544, 0.64562998], [0.47930001, 0.64838011], [0.45466267, 0.64061598]], [[0.50887558, 0.39900422], [0.51721343, 0.39245871], [0.50348963, 0.40090425], [0.49889359, 0.4035989], [0.50523872, 0.39916877]]], [[[0.26092216, 0.41247222], [0.26092216, 0.41247222], [0.26092216, 0.41247222], [0.26092216, 0.41247222], [0.26092216, 0.41247222]], [[0.34745133, 0.05888009], [0.34745133, 0.05888009], [0.34745133, 0.05888009], [0.34745133, 0.05888009], [0.34745133, 0.05888009]]]]
            tensor = transformer_units.dot_product_attention(q, k, v, 1.0, padding)
            session.run(tf.global_variables_initializer())
            actual = session.run(tensor)
            self.assertAllClose(actual, expected, 1e-06, 1e-06)
if __name__ == '__main__':
    googletest.main()