"""Test Transformer model helper methods."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from official.transformer.model import model_utils
from official.utils.misc import keras_utils
NEG_INF = -1000000000.0

class ModelUtilsTest(tf.test.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(ModelUtilsTest, self).setUp()
        if keras_utils.is_v2_0:
            tf.compat.v1.disable_eager_execution()

    def test_get_padding(self):
        if False:
            i = 10
            return i + 15
        x = tf.constant([[1, 0, 0, 0, 2], [3, 4, 0, 0, 0], [0, 5, 6, 0, 7]])
        padding = model_utils.get_padding(x, padding_value=0)
        with self.session() as sess:
            padding = sess.run(padding)
        self.assertAllEqual([[0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [1, 0, 0, 1, 0]], padding)

    def test_get_padding_bias(self):
        if False:
            for i in range(10):
                print('nop')
        x = tf.constant([[1, 0, 0, 0, 2], [3, 4, 0, 0, 0], [0, 5, 6, 0, 7]])
        bias = model_utils.get_padding_bias(x)
        bias_shape = tf.shape(bias)
        flattened_bias = tf.reshape(bias, [3, 5])
        with self.session() as sess:
            (flattened_bias, bias_shape) = sess.run((flattened_bias, bias_shape))
        self.assertAllEqual([[0, NEG_INF, NEG_INF, NEG_INF, 0], [0, 0, NEG_INF, NEG_INF, NEG_INF], [NEG_INF, 0, 0, NEG_INF, 0]], flattened_bias)
        self.assertAllEqual([3, 1, 1, 5], bias_shape)

    def test_get_decoder_self_attention_bias(self):
        if False:
            return 10
        length = 5
        bias = model_utils.get_decoder_self_attention_bias(length)
        with self.session() as sess:
            bias = sess.run(bias)
        self.assertAllEqual([[[[0, NEG_INF, NEG_INF, NEG_INF, NEG_INF], [0, 0, NEG_INF, NEG_INF, NEG_INF], [0, 0, 0, NEG_INF, NEG_INF], [0, 0, 0, 0, NEG_INF], [0, 0, 0, 0, 0]]]], bias)
if __name__ == '__main__':
    tf.test.main()