import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common

class TestModule(tf.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(TestModule, self).__init__()
        self.v = tf.Variable(42.0)

    @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
    def callee(self, x):
        if False:
            for i in range(10):
                print('nop')
        return (x, self.v)

    @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
    def caller(self, x):
        if False:
            while True:
                i = 10
        return self.callee(x)
if __name__ == '__main__':
    common.do_test(TestModule)