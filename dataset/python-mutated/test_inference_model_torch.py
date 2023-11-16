import os
import pytest
from unittest import TestCase
from bigdl.orca.inference import InferenceModel
from bigdl.orca.torch import zoo_pickle_module
import torch
import torchvision
from bigdl.dllib.nncontext import *

class TestInferenceModelTorch(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        ' setup any state tied to the execution of the given method in a\n        class.  setup_method is invoked for every test method of a class.\n        '
        self.sc = init_spark_on_local(4)

    def tearDown(self):
        if False:
            while True:
                i = 10
        ' teardown any state that was previously setup with a setup_method\n        call.\n        '
        self.sc.stop()

    def test_load_torch(self):
        if False:
            while True:
                i = 10
        torch_model = torchvision.models.resnet18()
        tmp_path = create_tmp_path() + '.pt'
        torch.save(torch_model, tmp_path, pickle_module=zoo_pickle_module)
        model = InferenceModel(10)
        model.load_torch(tmp_path)
        input_data = np.random.random([4, 3, 224, 224])
        output_data = model.predict(input_data)
        os.remove(tmp_path)
if __name__ == '__main__':
    pytest.main([__file__])