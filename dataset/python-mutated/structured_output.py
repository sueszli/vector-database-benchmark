import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common

class TestModule(tf.Module):

    @tf.function(input_signature=[])
    def f0000_single_return(self):
        if False:
            return 10
        return tf.constant(1.0, shape=[1])

    @tf.function(input_signature=[])
    def f0001_multiple_results_no_punctuation(self):
        if False:
            print('Hello World!')
        return (tf.constant(1.0, shape=[1]), tf.constant(1.0, shape=[2]))

    @tf.function(input_signature=[])
    def f0002_multiple_results_parentheses(self):
        if False:
            while True:
                i = 10
        return (tf.constant(1.0, shape=[1]), tf.constant(1.0, shape=[2]))

    @tf.function(input_signature=[])
    def f0003_multiple_results_brackets(self):
        if False:
            for i in range(10):
                print('nop')
        return [tf.constant(1.0, shape=[1]), tf.constant(1.0, shape=[2])]

    @tf.function(input_signature=[])
    def f0004_list_2_elements(self):
        if False:
            while True:
                i = 10
        return [[tf.constant(1.0, shape=[1]), tf.constant(1.0, shape=[2])]]

    @tf.function(input_signature=[])
    def f0005_dict_2_keys(self):
        if False:
            while True:
                i = 10
        return {'x': tf.constant(1.0, shape=[1]), 'y': tf.constant(1.0, shape=[2])}

    @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
    def f0006_multiple_return_statements(self, x):
        if False:
            return 10
        if x > 3.0:
            return {'x': tf.constant(1.0, shape=[1])}
        else:
            return {'x': tf.constant(1.0, shape=[1])}
if __name__ == '__main__':
    common.do_test(TestModule)