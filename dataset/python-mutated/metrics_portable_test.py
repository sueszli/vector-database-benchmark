"""TensorFlow Lite Python metrics helpr TFLiteMetrics check."""
from tensorflow.lite.python.metrics import metrics
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class MetricsPortableTest(test_util.TensorFlowTestCase):

    def test_TFLiteMetrics_creation_success(self):
        if False:
            return 10
        metrics.TFLiteMetrics()

    def test_debugger_creation_counter_increase_success(self):
        if False:
            return 10
        stub = metrics.TFLiteMetrics()
        stub.increase_counter_debugger_creation()

    def test_interpreter_creation_counter_increase_success(self):
        if False:
            i = 10
            return i + 15
        stub = metrics.TFLiteMetrics()
        stub.increase_counter_interpreter_creation()

    def test_converter_attempt_counter_increase_success(self):
        if False:
            for i in range(10):
                print('nop')
        stub = metrics.TFLiteMetrics()
        stub.increase_counter_converter_attempt()

    def test_converter_success_counter_increase_success(self):
        if False:
            print('Hello World!')
        stub = metrics.TFLiteMetrics()
        stub.increase_counter_converter_success()

    def test_converter_params_set_success(self):
        if False:
            while True:
                i = 10
        stub = metrics.TFLiteMetrics()
        stub.set_converter_param('name', 'value')
if __name__ == '__main__':
    test.main()