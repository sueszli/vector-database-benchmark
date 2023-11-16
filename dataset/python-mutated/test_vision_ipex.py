import pytest
import os
from unittest import TestCase
from bigdl.nano.pytorch.vision.models import vision
from test.pytorch.utils._train_torch_lightning import train_with_linear_top_layer
from bigdl.nano.utils.pytorch import TORCH_VERSION_LESS_2_0
from bigdl.nano.utils.common import _avx2_checker
batch_size = 256
num_workers = 0
data_dir = '/tmp/data'

class VisionIPEX:

    def test_resnet18_ipex(self):
        if False:
            i = 10
            return i + 15
        resnet18 = vision.resnet18(pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(resnet18, batch_size, num_workers, data_dir, use_ipex=True)

    def test_resnet34_ipex(self):
        if False:
            for i in range(10):
                print('nop')
        resnet34 = vision.resnet34(pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(resnet34, batch_size, num_workers, data_dir, use_ipex=True)

    def test_resnet50_ipex(self):
        if False:
            while True:
                i = 10
        resnet50 = vision.resnet50(pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(resnet50, batch_size, num_workers, data_dir, use_ipex=True)

    def test_mobilenet_v3_large_ipex(self):
        if False:
            while True:
                i = 10
        mobilenet = vision.mobilenet_v3_large(pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(mobilenet, batch_size, num_workers, data_dir, use_ipex=True)

    def test_mobilenet_v3_small_ipex(self):
        if False:
            return 10
        mobilenet = vision.mobilenet_v3_small(pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(mobilenet, batch_size, num_workers, data_dir, use_ipex=True)

    def test_mobilenet_v2_ipex(self):
        if False:
            return 10
        mobilenet = vision.mobilenet_v2(pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(mobilenet, batch_size, num_workers, data_dir, use_ipex=True)

    def test_shufflenet_ipex(self):
        if False:
            while True:
                i = 10
        shufflenet = vision.shufflenet_v2_x1_0(pretrained=False, include_top=False, freeze=True)
        train_with_linear_top_layer(shufflenet, batch_size, num_workers, data_dir, use_ipex=True)
TORCH_CLS = VisionIPEX

class CaseWithoutAVX2:

    def test_placeholder(self):
        if False:
            i = 10
            return i + 15
        pass
if not TORCH_VERSION_LESS_2_0 and (not _avx2_checker()):
    print('Vision IPEX Without AVX2')
    TORCH_CLS = CaseWithoutAVX2

class TestVisionIPEX(TORCH_CLS, TestCase):
    pass
if __name__ == '__main__':
    pytest.main([__file__])