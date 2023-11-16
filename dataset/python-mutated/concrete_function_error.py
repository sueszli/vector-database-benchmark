"""Test file to display the error message and verify it with FileCheck."""
import sys
from absl import app
import tensorflow.compat.v2 as tf
if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()

class TestGraphDebugInfo(object):
    """Test stack trace can be displayed."""

    def testConcreteFunctionDebugInfo(self):
        if False:
            return 10
        'Create a concrete func with unsupported ops, and convert it.'

        @tf.function(input_signature=[tf.TensorSpec(shape=[3, 3], dtype=tf.float32)])
        def model(x):
            if False:
                print('Hello World!')
            y = tf.math.betainc(x, 0.5, 1.0)
            return y + y
        func = model.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([func], model)
        converter.convert()

def main(argv):
    if False:
        return 10
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    try:
        TestGraphDebugInfo().testConcreteFunctionDebugInfo()
    except Exception as e:
        sys.stdout.write('testConcreteFunctionDebugInfo')
        sys.stdout.write(str(e))
if __name__ == '__main__':
    app.run(main)