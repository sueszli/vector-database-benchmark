import os
import shutil
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope.models.base import TorchModel
from modelscope.preprocessors import Preprocessor
from modelscope.utils.regress_test_utils import compare_arguments_nested, numpify_tensor_nested

class TorchBaseTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        print('Testing %s.%s' % (type(self).__name__, self._testMethodName))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        if False:
            while True:
                i = 10
        shutil.rmtree(self.tmp_dir)
        super().tearDown()

    def test_custom_model(self):
        if False:
            while True:
                i = 10

        class MyTorchModel(TorchModel):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                x = F.relu(self.conv1(input))
                return F.relu(self.conv2(x))
        model = MyTorchModel()
        model.train()
        model.eval()
        out = model.forward(torch.rand(1, 1, 10, 10))
        self.assertEqual((1, 20, 2, 2), out.shape)

    def test_custom_model_with_postprocess(self):
        if False:
            print('Hello World!')
        add_bias = 200

        class MyTorchModel(TorchModel):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                x = F.relu(self.conv1(input))
                return F.relu(self.conv2(x))

            def postprocess(self, x):
                if False:
                    return 10
                return x + add_bias
        model = MyTorchModel()
        model.train()
        model.eval()
        out = model(torch.rand(1, 1, 10, 10))
        self.assertEqual((1, 20, 2, 2), out.shape)
        self.assertTrue(np.all(out.detach().numpy() > add_bias - 10))

    def test_save_pretrained(self):
        if False:
            return 10
        preprocessor = Preprocessor.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-tiny')
        model = TorchModel.from_pretrained('damo/nlp_structbert_sentence-similarity_chinese-tiny')
        model.eval()
        with torch.no_grad():
            res1 = numpify_tensor_nested(model(**preprocessor(('test1', 'test2'))))
        save_path = os.path.join(self.tmp_dir, 'test_save_pretrained')
        model.save_pretrained(save_path, save_checkpoint_names='pytorch_model.bin')
        self.assertTrue(os.path.isfile(os.path.join(save_path, 'pytorch_model.bin')))
        self.assertTrue(os.path.isfile(os.path.join(save_path, 'configuration.json')))
        self.assertTrue(os.path.isfile(os.path.join(save_path, 'vocab.txt')))
        model = TorchModel.from_pretrained(save_path)
        model.eval()
        with torch.no_grad():
            res2 = numpify_tensor_nested(model(**preprocessor(('test1', 'test2'))))
        self.assertTrue(compare_arguments_nested('', res1, res2))
if __name__ == '__main__':
    unittest.main()