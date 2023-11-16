import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common

class Child(tf.Module):

    def __init__(self):
        if False:
            return 10
        super(Child, self).__init__()
        self.my_variable = tf.Variable(3.0)

class TestModule(tf.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(TestModule, self).__init__()
        self.child1 = Child()
        self.child2 = self.child1
if __name__ == '__main__':
    common.do_test(TestModule)