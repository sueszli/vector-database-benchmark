import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common

class TestModule(tf.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(TestModule, self).__init__()
        self.v0 = tf.Variable([0.0], shape=tf.TensorShape(None))
        self.v1 = tf.Variable([0.0, 1.0], shape=[None])
if __name__ == '__main__':
    common.do_test(TestModule, exported_names=[])