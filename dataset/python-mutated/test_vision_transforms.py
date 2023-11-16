import pytest
import os
from unittest import TestCase
from bigdl.nano.pytorch.vision.models import vision
from test.pytorch.utils._train_torch_lightning import train_with_linear_top_layer
batch_size = 256
num_workers = 0
data_dir = '/tmp/data'

class TestVisionTransforms(TestCase):

    def test_if_transforms_missing(self):
        if False:
            while True:
                i = 10
        from bigdl.nano.pytorch.vision.transforms import transforms as nano_module
        from torchvision.transforms import transforms as compared_module
        assert all([x in nano_module.__all__ for x in compared_module.__all__]), 'Missing transforms'
if __name__ == '__main__':
    pytest.main([__file__])