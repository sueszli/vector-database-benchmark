"""Test for version 1 of the zero_out op."""
import tensorflow as tf
from tensorflow.examples.adding_an_op import cuda_op

class AddOneTest(tf.test.TestCase):

    def test(self):
        if False:
            print('Hello World!')
        if tf.test.is_built_with_cuda():
            result = cuda_op.add_one([5, 4, 3, 2, 1])
            self.assertAllEqual(result, [6, 5, 4, 3, 2])
if __name__ == '__main__':
    tf.test.main()