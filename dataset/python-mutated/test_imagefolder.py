import pytest
import os
from unittest import TestCase
from bigdl.nano.pytorch.vision.models import vision
from test.pytorch.utils._train_imagefolder import train_torch_lightning
batch_size = 1
resources_root = os.path.join(os.path.dirname(__file__), '../../resources')
root_dir1 = os.path.join(resources_root, 'train_image_folder_png')
root_dir2 = os.path.join(resources_root, 'train_image_folder_jpg')

class TestImageFolder(TestCase):

    def test_resnet18_quantitrain_image_folder_pngze(self):
        if False:
            for i in range(10):
                print('nop')
        resnet18 = vision.resnet18(pretrained=False, include_top=False, freeze=True)
        train_torch_lightning(resnet18, root_dir1, batch_size)
        train_torch_lightning(resnet18, root_dir2, batch_size)
if __name__ == '__main__':
    pytest.main([__file__])