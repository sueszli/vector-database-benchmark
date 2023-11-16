"""TFLite metrics_wrapper module test cases."""
import tensorflow as tf
from tensorflow.lite.python import lite
from tensorflow.lite.python.convert import ConverterError
from tensorflow.lite.python.metrics.wrapper import metrics_wrapper
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class MetricsWrapperTest(test_util.TensorFlowTestCase):

    def test_basic_retrieve_collected_errors_empty(self):
        if False:
            for i in range(10):
                print('nop')
        errors = metrics_wrapper.retrieve_collected_errors()
        self.assertEmpty(errors)

    def test_basic_retrieve_collected_errors_not_empty(self):
        if False:
            i = 10
            return i + 15

        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def func(x):
            if False:
                return 10
            return tf.cosh(x)
        converter = lite.TFLiteConverterV2.from_concrete_functions([func.get_concrete_function()], func)
        try:
            converter.convert()
        except ConverterError as err:
            captured_errors = err.errors
        self.assertNotEmpty(captured_errors)
if __name__ == '__main__':
    test.main()