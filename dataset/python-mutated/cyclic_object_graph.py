import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common

class ReferencesParent(tf.Module):

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        super(ReferencesParent, self).__init__()
        self.parent = parent
        self.my_variable = tf.Variable(3.0)

class TestModule(tf.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestModule, self).__init__()
        self.child = ReferencesParent(self)
if __name__ == '__main__':
    common.do_test(TestModule)