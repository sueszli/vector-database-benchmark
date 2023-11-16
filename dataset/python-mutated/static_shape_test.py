"""Tests for object_detection.utils.static_shape."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from object_detection.utils import static_shape

class StaticShapeTest(tf.test.TestCase):

    def test_return_correct_batchSize(self):
        if False:
            i = 10
            return i + 15
        tensor_shape = tf.TensorShape(dims=[32, 299, 384, 3])
        self.assertEqual(32, static_shape.get_batch_size(tensor_shape))

    def test_return_correct_height(self):
        if False:
            for i in range(10):
                print('nop')
        tensor_shape = tf.TensorShape(dims=[32, 299, 384, 3])
        self.assertEqual(299, static_shape.get_height(tensor_shape))

    def test_return_correct_width(self):
        if False:
            while True:
                i = 10
        tensor_shape = tf.TensorShape(dims=[32, 299, 384, 3])
        self.assertEqual(384, static_shape.get_width(tensor_shape))

    def test_return_correct_depth(self):
        if False:
            while True:
                i = 10
        tensor_shape = tf.TensorShape(dims=[32, 299, 384, 3])
        self.assertEqual(3, static_shape.get_depth(tensor_shape))

    def test_die_on_tensor_shape_with_rank_three(self):
        if False:
            for i in range(10):
                print('nop')
        tensor_shape = tf.TensorShape(dims=[32, 299, 384])
        with self.assertRaises(ValueError):
            static_shape.get_batch_size(tensor_shape)
            static_shape.get_height(tensor_shape)
            static_shape.get_width(tensor_shape)
            static_shape.get_depth(tensor_shape)
if __name__ == '__main__':
    tf.test.main()