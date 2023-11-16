"""Tests for tf upgrader."""
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
_TEST_VERSION = 1

class TestUpgrade(test_util.TensorFlowTestCase):
    """Test various APIs that have been changed in 2.0."""

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        cls._tf_api_version = 1 if hasattr(tf, 'contrib') else 2

    def setUp(self):
        if False:
            i = 10
            return i + 15
        tf.compat.v1.enable_v2_behavior()

    def testRenames(self):
        if False:
            return 10
        self.assertAllClose(1.04719755, tf.acos(0.5))
        self.assertAllClose(0.5, tf.rsqrt(4.0))

    def testSerializeSparseTensor(self):
        if False:
            for i in range(10):
                print('nop')
        sp_input = tf.SparseTensor(indices=tf.constant([[1]], dtype=tf.int64), values=tf.constant([2], dtype=tf.int64), dense_shape=[2])
        with self.cached_session():
            serialized_sp = tf.serialize_sparse(sp_input, 'serialize_name', tf.string)
            self.assertEqual((3,), serialized_sp.shape)
            self.assertTrue(serialized_sp[0].numpy())

    def testSerializeManySparse(self):
        if False:
            while True:
                i = 10
        sp_input = tf.SparseTensor(indices=tf.constant([[0, 1]], dtype=tf.int64), values=tf.constant([2], dtype=tf.int64), dense_shape=[1, 2])
        with self.cached_session():
            serialized_sp = tf.serialize_many_sparse(sp_input, 'serialize_name', tf.string)
            self.assertEqual((1, 3), serialized_sp.shape)

    def testArgMaxMin(self):
        if False:
            i = 10
            return i + 15
        self.assertAllClose([1], tf.argmax([[1, 3, 2]], name='abc', dimension=1))
        self.assertAllClose([0, 0, 0], tf.argmax([[1, 3, 2]], dimension=0))
        self.assertAllClose([0], tf.argmin([[1, 3, 2]], name='abc', dimension=1))

    def testSoftmaxCrossEntropyWithLogits(self):
        if False:
            while True:
                i = 10
        out = tf.nn.softmax_cross_entropy_with_logits(logits=[0.1, 0.8], labels=[0, 1])
        self.assertAllClose(out, 0.40318608)
        out = tf.nn.softmax_cross_entropy_with_logits_v2(logits=[0.1, 0.8], labels=[0, 1])
        self.assertAllClose(out, 0.40318608)
if __name__ == '__main__':
    test_lib.main()