"""CodeLab for displaying error stack trace w/ MLIR-based converter."""
import sys
from absl import app
import tensorflow as tf

def suppress_exception(f):
    if False:
        for i in range(10):
            print('nop')

    def wrapped():
        if False:
            while True:
                i = 10
        try:
            f()
        except:
            pass
    return wrapped

class TestModule(tf.Module):
    """The test model has unsupported op."""

    @tf.function(input_signature=[tf.TensorSpec(shape=[3, 3], dtype=tf.float32)])
    def model(self, x):
        if False:
            print('Hello World!')
        y = tf.math.reciprocal(x)
        return y + y

@suppress_exception
def test_from_saved_model():
    if False:
        while True:
            i = 10
    'displaying stack trace when converting saved model.'
    test_model = TestModule()
    saved_model_path = '/tmp/test.saved_model'
    save_options = tf.saved_model.SaveOptions(save_debug_info=True)
    tf.saved_model.save(test_model, saved_model_path, options=save_options)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.convert()

def test_from_concrete_function():
    if False:
        for i in range(10):
            print('nop')
    'displaying stack trace when converting concrete function.'

    @tf.function(input_signature=[tf.TensorSpec(shape=[3, 3], dtype=tf.float32)])
    def model(x):
        if False:
            while True:
                i = 10
        y = tf.math.reciprocal(x)
        return y + y
    func = model.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([func], model)
    converter.convert()

def main(argv):
    if False:
        return 10
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    sys.stdout.write('==== Testing from_concrete_functions ====\n')
    test_from_concrete_function()
    sys.stdout.write('==== Testing from_saved_model ====\n')
    test_from_saved_model()
if __name__ == '__main__':
    app.run(main)