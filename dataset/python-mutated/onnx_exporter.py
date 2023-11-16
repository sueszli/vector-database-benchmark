import os
import onnx
import torch
from ludwig.api import LudwigModel
from ludwig.model_export.base_model_exporter import BaseModelExporter, LudwigTorchWrapper

class OnnxExporter(BaseModelExporter):
    """Class that abstracts the convertion of torch to onnx."""

    def export(self, model_path, export_path, output_model_name):
        if False:
            for i in range(10):
                print('nop')
        ludwig_model = LudwigModel.load(model_path)
        model = LudwigTorchWrapper(ludwig_model.model)
        model.eval()
        width = ludwig_model.config['input_features'][0]['preprocessing']['width']
        height = ludwig_model.config['input_features'][0]['preprocessing']['height']
        example_input = torch.randn(1, 3, width, height, requires_grad=True)
        torch.onnx.export(model, example_input, os.path.join(export_path, output_model_name), opset_version=18, export_params=True, do_constant_folding=True, input_names=['input'], output_names=['combiner_hidden_1', 'output', 'combiner_hidden_2'])

    def check_model_export(self, path):
        if False:
            i = 10
            return i + 15
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)