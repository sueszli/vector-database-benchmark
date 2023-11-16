import logging
from apache_beam.examples.inference import pytorch_image_classification
from apache_beam.testing.load_tests.load_test import LoadTest
from torchvision import models
_PERF_TEST_MODELS = ['resnet50', 'resnet101', 'resnet152']
_PRETRAINED_MODEL_MODULE = 'torchvision.models'

class PytorchVisionBenchmarkTest(LoadTest):

    def __init__(self):
        if False:
            print('Hello World!')
        self.metrics_namespace = 'BeamML_PyTorch'
        super().__init__(metrics_namespace=self.metrics_namespace)

    def test(self):
        if False:
            return 10
        pretrained_model_name = self.pipeline.get_option('pretrained_model_name')
        if not pretrained_model_name:
            raise RuntimeError('Please provide a pretrained torch model name. Model name must be from the module torchvision.models')
        if pretrained_model_name == _PERF_TEST_MODELS[0]:
            model_class = models.resnet50
        elif pretrained_model_name == _PERF_TEST_MODELS[1]:
            model_class = models.resnet101
        elif pretrained_model_name == _PERF_TEST_MODELS[2]:
            model_class = models.resnet152
        else:
            raise NotImplementedError
        model_params = {'num_classes': 1000, 'pretrained': False}
        extra_opts = {}
        extra_opts['input'] = self.pipeline.get_option('input_file')
        device = self.pipeline.get_option('device')
        self.result = pytorch_image_classification.run(self.pipeline.get_full_options_as_args(**extra_opts), model_class=model_class, model_params=model_params, test_pipeline=self.pipeline, device=device)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    PytorchVisionBenchmarkTest().run()