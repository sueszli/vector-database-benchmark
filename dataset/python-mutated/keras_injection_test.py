"""Keras injection tests."""
import tensorflow as tf
from tensorflow.python.eager import test

class KerasInjectionTest(tf.test.TestCase):

    def test_keras_optimizer_injected(self):
        if False:
            print('Hello World!')
        save_path = test.test_src_dir_path('cc/saved_model/testdata/OptimizerSlotVariableModule')
        _ = tf.saved_model.load(save_path)
        self.assertIn('optimizer', tf.__internal__.saved_model.load.registered_identifiers())
if __name__ == '__main__':
    tf.test.main()