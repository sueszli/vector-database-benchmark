import os
import unittest
import onnx_test_common
import parameterized
import PIL
import torch
import torchvision
from torch import nn

def _get_test_image_tensor():
    if False:
        i = 10
        return i + 15
    data_dir = os.path.join(os.path.dirname(__file__), 'assets')
    img_path = os.path.join(data_dir, 'grace_hopper_517x606.jpg')
    input_image = PIL.Image.open(img_path)
    preprocess = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return preprocess(input_image).unsqueeze(0)

class _TopPredictor(nn.Module):

    def __init__(self, base_model):
        if False:
            while True:
                i = 10
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.base_model(x)
        (_, topk_id) = torch.topk(x[0], 1)
        return topk_id

@parameterized.parameterized_class(('is_script',), [(True,), (False,)], class_name_func=onnx_test_common.parameterize_class_name)
class TestQuantizedModelsONNXRuntime(onnx_test_common._TestONNXRuntime):

    def run_test(self, model, inputs, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        model = _TopPredictor(model)
        return super().run_test(model, inputs, *args, **kwargs)

    def test_mobilenet_v3(self):
        if False:
            print('Hello World!')
        model = torchvision.models.quantization.mobilenet_v3_large(pretrained=True, quantize=True)
        self.run_test(model, _get_test_image_tensor())

    @unittest.skip('quantized::cat not supported')
    def test_inception_v3(self):
        if False:
            while True:
                i = 10
        model = torchvision.models.quantization.inception_v3(pretrained=True, quantize=True)
        self.run_test(model, _get_test_image_tensor())

    @unittest.skip('quantized::cat not supported')
    def test_googlenet(self):
        if False:
            while True:
                i = 10
        model = torchvision.models.quantization.googlenet(pretrained=True, quantize=True)
        self.run_test(model, _get_test_image_tensor())

    @unittest.skip('quantized::cat not supported')
    def test_shufflenet_v2_x0_5(self):
        if False:
            while True:
                i = 10
        model = torchvision.models.quantization.shufflenet_v2_x0_5(pretrained=True, quantize=True)
        self.run_test(model, _get_test_image_tensor())

    def test_resnet18(self):
        if False:
            while True:
                i = 10
        model = torchvision.models.quantization.resnet18(pretrained=True, quantize=True)
        self.run_test(model, _get_test_image_tensor())

    def test_resnet50(self):
        if False:
            return 10
        model = torchvision.models.quantization.resnet50(pretrained=True, quantize=True)
        self.run_test(model, _get_test_image_tensor())

    def test_resnext101_32x8d(self):
        if False:
            return 10
        model = torchvision.models.quantization.resnext101_32x8d(pretrained=True, quantize=True)
        self.run_test(model, _get_test_image_tensor())