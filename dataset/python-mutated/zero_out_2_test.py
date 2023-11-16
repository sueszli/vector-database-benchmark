"""Test for version 2 of the zero_out op."""
import tensorflow as tf
from tensorflow.examples.adding_an_op import zero_out_grad_2
from tensorflow.examples.adding_an_op import zero_out_op_2

class ZeroOut2Test(tf.test.TestCase):

    def test(self):
        if False:
            while True:
                i = 10
        result = zero_out_op_2.zero_out([5, 4, 3, 2, 1])
        self.assertAllEqual(result, [5, 0, 0, 0, 0])

    def test_2d(self):
        if False:
            return 10
        result = zero_out_op_2.zero_out([[6, 5, 4], [3, 2, 1]])
        self.assertAllEqual(result, [[6, 0, 0], [0, 0, 0]])

    def test_grad(self):
        if False:
            while True:
                i = 10
        x = tf.constant([5, 4, 3, 2, 1], dtype=tf.float32)
        (theoretical, numerical) = tf.test.compute_gradient(zero_out_op_2.zero_out, tuple([x]))
        self.assertAllClose(theoretical, numerical)

    def test_grad_2d(self):
        if False:
            return 10
        x = tf.constant([[6, 5, 4], [3, 2, 1]], dtype=tf.float32)
        (theoretical, numerical) = tf.test.compute_gradient(zero_out_op_2.zero_out, tuple([x]))
        self.assertAllClose(theoretical, numerical)
if __name__ == '__main__':
    tf.test.main()