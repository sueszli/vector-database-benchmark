"""Test file to display the error message and verify it with FileCheck."""
import sys
from absl import app
import tensorflow.compat.v2 as tf
if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()

class TestModule(tf.Module):
    """The test model has unsupported op."""

    @tf.function(input_signature=[tf.TensorSpec(shape=[3, 3], dtype=tf.float32)])
    def model(self, x):
        if False:
            i = 10
            return i + 15
        y = tf.math.betainc(x, 0.5, 1.0)
        return y + y

class TestGraphDebugInfo(object):
    """Test stack trace can be displayed."""

    def testSavedModelDebugInfo(self):
        if False:
            return 10
        'Save a saved model with unsupported ops, and then load and convert it.'
        test_model = TestModule()
        saved_model_path = '/tmp/test.saved_model'
        save_options = tf.saved_model.SaveOptions(save_debug_info=True)
        tf.saved_model.save(test_model, saved_model_path, options=save_options)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.convert()

def main(argv):
    if False:
        print('Hello World!')
    'test driver method writes the error message to stdout.'
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    try:
        TestGraphDebugInfo().testSavedModelDebugInfo()
    except Exception as e:
        sys.stdout.write('testSavedModelDebugInfo')
        sys.stdout.write(str(e))
if __name__ == '__main__':
    app.run(main)