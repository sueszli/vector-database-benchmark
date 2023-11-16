import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common

class TestModule(tf.Module):

    @tf.function(input_signature=[tf.TensorSpec([1], tf.float32), tf.TensorSpec([2], tf.float32)])
    def f0000_function_arity(self, x, y):
        if False:
            i = 10
            return i + 15
        return

    @tf.function(input_signature=[[tf.TensorSpec([], tf.float32), tf.TensorSpec([], tf.float32)]])
    def f0001_list_2_elements(self, l):
        if False:
            i = 10
            return i + 15
        return

    @tf.function(input_signature=[{'x': tf.TensorSpec([1], tf.float32), 'y': tf.TensorSpec([2], tf.float32)}])
    def f0002_dict_2_keys(self, d):
        if False:
            while True:
                i = 10
        return

    @tf.function(input_signature=[{'y': tf.TensorSpec([2], tf.float32), 'x': tf.TensorSpec([1], tf.float32)}])
    def f0003_dict_2_keys_out_of_order(self, d):
        if False:
            for i in range(10):
                print('nop')
        return

    @tf.function(input_signature=[{'x': tf.TensorSpec([4], tf.float32), 'y': tf.TensorSpec([5], tf.float32), 'z': tf.TensorSpec([6], tf.float32), 'a': tf.TensorSpec([1], tf.float32), 'b': tf.TensorSpec([2], tf.float32), 'c': tf.TensorSpec([3], tf.float32)}])
    def f0004_dict_many_keys(self, d):
        if False:
            while True:
                i = 10
        return

    @tf.function(input_signature=[{'x': [tf.TensorSpec([1], tf.float32), tf.TensorSpec([2], tf.float32)], 'y': tf.TensorSpec([3], tf.float32)}])
    def f0005_more_complex_recursive_structure(self, d):
        if False:
            print('Hello World!')
        return
if __name__ == '__main__':
    common.do_test(TestModule)