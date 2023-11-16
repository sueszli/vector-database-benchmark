"""Test for version 3 of the zero_out op."""
import tensorflow as tf
from tensorflow.examples.adding_an_op import zero_out_op_3

class ZeroOut3Test(tf.test.TestCase):

    def test(self):
        if False:
            i = 10
            return i + 15
        result = zero_out_op_3.zero_out([5, 4, 3, 2, 1])
        self.assertAllEqual(result, [5, 0, 0, 0, 0])

    def test_attr(self):
        if False:
            print('Hello World!')
        result = zero_out_op_3.zero_out([5, 4, 3, 2, 1], preserve_index=3)
        self.assertAllEqual(result, [0, 0, 0, 2, 0])

    def test_negative(self):
        if False:
            print('Hello World!')
        with self.assertRaisesOpError('Need preserve_index >= 0, got -1'):
            self.evaluate(zero_out_op_3.zero_out([5, 4, 3, 2, 1], preserve_index=-1))

    def test_large(self):
        if False:
            print('Hello World!')
        with self.assertRaisesOpError('preserve_index out of range'):
            self.evaluate(zero_out_op_3.zero_out([5, 4, 3, 2, 1], preserve_index=17))
if __name__ == '__main__':
    tf.test.main()