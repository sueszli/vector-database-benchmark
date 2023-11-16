import logging
from apache_beam.examples.inference import pytorch_language_modeling
from apache_beam.testing.load_tests.load_test import LoadTest

class PytorchLanguageModelingBenchmarkTest(LoadTest):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.metrics_namespace = 'BeamML_PyTorch'
        super().__init__(metrics_namespace=self.metrics_namespace)

    def test(self):
        if False:
            i = 10
            return i + 15
        extra_opts = {}
        extra_opts['input'] = self.pipeline.get_option('input_file')
        self.result = pytorch_language_modeling.run(self.pipeline.get_full_options_as_args(**extra_opts), test_pipeline=self.pipeline)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    PytorchLanguageModelingBenchmarkTest().run()