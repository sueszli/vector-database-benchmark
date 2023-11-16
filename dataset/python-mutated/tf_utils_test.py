"""Tests for tensorflow_hub.tf_utils."""
import tensorflow as tf
from tensorflow_hub import tf_utils

class TfUtilsTest(tf.test.TestCase):

    def testIsCompositeTensor(self):
        if False:
            for i in range(10):
                print('nop')
        ragged_tensor = tf.ragged.constant([[1, 2], [3]])
        self.assertTrue(tf_utils.is_composite_tensor(ragged_tensor))
        sparse_tensor = tf.SparseTensor([[0, 2], [3, 2]], [5, 6], [10, 10])
        self.assertTrue(tf_utils.is_composite_tensor(sparse_tensor))
        tensor = tf.constant([1, 2, 3])
        self.assertFalse(tf_utils.is_composite_tensor(tensor))

    def testGetCompositeTensorTypeSpec(self):
        if False:
            while True:
                i = 10
        ragged_tensor = tf.ragged.constant([[1, 2], [3]])
        self.assertIsInstance(tf_utils.get_composite_tensor_type_spec(ragged_tensor), tf.RaggedTensorSpec)
        sparse_tensor = tf.SparseTensor([[0, 2], [3, 2]], [5, 6], [10, 10])
        self.assertIsInstance(tf_utils.get_composite_tensor_type_spec(sparse_tensor), tf.SparseTensorSpec)
        tensor = tf.constant([1, 2, 3])
        self.assertIs(tf_utils.get_composite_tensor_type_spec(tensor), None)
if __name__ == '__main__':
    tf.test.main()