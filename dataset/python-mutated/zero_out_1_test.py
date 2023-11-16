"""Test for version 1 of the zero_out op."""
import os.path
import tensorflow as tf
from tensorflow.examples.adding_an_op import zero_out_op_1

class ZeroOut1Test(tf.test.TestCase):

    def test(self):
        if False:
            return 10
        result = zero_out_op_1.zero_out([5, 4, 3, 2, 1])
        self.assertAllEqual(result, [5, 0, 0, 0, 0])

    def test_namespace(self):
        if False:
            print('Hello World!')
        result = zero_out_op_1.namespace_zero_out([5, 4, 3, 2, 1])
        self.assertAllEqual(result, [5, 0, 0, 0, 0])

    def test_namespace_call_op_on_op(self):
        if False:
            return 10
        x = zero_out_op_1.namespace_zero_out([5, 4, 3, 2, 1])
        result = zero_out_op_1.namespace_zero_out(x)
        self.assertAllEqual(result, [5, 0, 0, 0, 0])

    def test_namespace_nested(self):
        if False:
            for i in range(10):
                print('nop')
        result = zero_out_op_1.namespace_nested_zero_out([5, 4, 3, 2, 1])
        self.assertAllEqual(result, [5, 0, 0, 0, 0])

    def test_load_twice(self):
        if False:
            for i in range(10):
                print('nop')
        zero_out_loaded_again = tf.load_op_library(os.path.join(tf.compat.v1.resource_loader.get_data_files_path(), 'zero_out_op_kernel_1.so'))
        self.assertEqual(zero_out_loaded_again, zero_out_op_1._zero_out_module)
if __name__ == '__main__':
    tf.test.main()