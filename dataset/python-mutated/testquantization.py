"""
Quantization module tests
"""
import platform
import unittest
from transformers import AutoModel
from txtai.pipeline import HFModel, HFPipeline

class TestQuantization(unittest.TestCase):
    """
    Quantization tests.
    """

    @unittest.skipIf(platform.system() == 'Darwin', 'Quantized models not supported on macOS')
    def testModel(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test quantizing a model through HFModel.\n        '
        model = HFModel(quantize=True, gpu=False)
        model = model.prepare(AutoModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2'))
        self.assertIsNotNone(model)

    @unittest.skipIf(platform.system() == 'Darwin', 'Quantized models not supported on macOS')
    def testPipeline(self):
        if False:
            while True:
                i = 10
        '\n        Test quantizing a model through HFPipeline.\n        '
        pipeline = HFPipeline('text-classification', 'google/bert_uncased_L-2_H-128_A-2', True, False)
        self.assertIsNotNone(pipeline)