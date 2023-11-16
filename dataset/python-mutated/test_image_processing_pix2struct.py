import unittest
import numpy as np
import requests
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available
from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs
if is_torch_available():
    import torch
    from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_11
else:
    is_torch_greater_or_equal_than_1_11 = False
if is_vision_available():
    from PIL import Image
    from transformers import Pix2StructImageProcessor

class Pix2StructImageProcessingTester(unittest.TestCase):

    def __init__(self, parent, batch_size=7, num_channels=3, image_size=18, min_resolution=30, max_resolution=400, size=None, do_normalize=True, do_convert_rgb=True, patch_size=None):
        if False:
            i = 10
            return i + 15
        size = size if size is not None else {'height': 20, 'width': 20}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.size = size
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb
        self.max_patches = [512, 1024, 2048, 4096]
        self.patch_size = patch_size if patch_size is not None else {'height': 16, 'width': 16}

    def prepare_image_processor_dict(self):
        if False:
            for i in range(10):
                print('nop')
        return {'do_normalize': self.do_normalize, 'do_convert_rgb': self.do_convert_rgb}

    def prepare_dummy_image(self):
        if False:
            for i in range(10):
                print('nop')
        img_url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg'
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        return raw_image

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        if False:
            return 10
        return prepare_image_inputs(batch_size=self.batch_size, num_channels=self.num_channels, min_resolution=self.min_resolution, max_resolution=self.max_resolution, equal_resolution=equal_resolution, numpify=numpify, torchify=torchify)

@unittest.skipIf(not is_torch_greater_or_equal_than_1_11, reason='`Pix2StructImageProcessor` requires `torch>=1.11.0`.')
@require_torch
@require_vision
class Pix2StructImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = Pix2StructImageProcessor if is_vision_available() else None

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.image_processor_tester = Pix2StructImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        if False:
            for i in range(10):
                print('nop')
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        if False:
            return 10
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processor, 'do_normalize'))
        self.assertTrue(hasattr(image_processor, 'do_convert_rgb'))

    def test_expected_patches(self):
        if False:
            i = 10
            return i + 15
        dummy_image = self.image_processor_tester.prepare_dummy_image()
        image_processor = self.image_processing_class(**self.image_processor_dict)
        max_patch = 2048
        inputs = image_processor(dummy_image, return_tensors='pt', max_patches=max_patch)
        self.assertTrue(torch.allclose(inputs.flattened_patches.mean(), torch.tensor(0.0606), atol=0.001, rtol=0.001))

    def test_call_pil(self):
        if False:
            for i in range(10):
                print('nop')
        image_processor = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)
        expected_hidden_dim = self.image_processor_tester.patch_size['height'] * self.image_processor_tester.patch_size['width'] * self.image_processor_tester.num_channels + 2
        for max_patch in self.image_processor_tester.max_patches:
            encoded_images = image_processor(image_inputs[0], return_tensors='pt', max_patches=max_patch).flattened_patches
            self.assertEqual(encoded_images.shape, (1, max_patch, expected_hidden_dim))
            encoded_images = image_processor(image_inputs, return_tensors='pt', max_patches=max_patch).flattened_patches
            self.assertEqual(encoded_images.shape, (self.image_processor_tester.batch_size, max_patch, expected_hidden_dim))

    def test_call_vqa(self):
        if False:
            i = 10
            return i + 15
        image_processor = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)
        expected_hidden_dim = self.image_processor_tester.patch_size['height'] * self.image_processor_tester.patch_size['width'] * self.image_processor_tester.num_channels + 2
        image_processor.is_vqa = True
        for max_patch in self.image_processor_tester.max_patches:
            with self.assertRaises(ValueError):
                encoded_images = image_processor(image_inputs[0], return_tensors='pt', max_patches=max_patch).flattened_patches
            dummy_text = 'Hello'
            encoded_images = image_processor(image_inputs[0], return_tensors='pt', max_patches=max_patch, header_text=dummy_text).flattened_patches
            self.assertEqual(encoded_images.shape, (1, max_patch, expected_hidden_dim))
            encoded_images = image_processor(image_inputs, return_tensors='pt', max_patches=max_patch, header_text=dummy_text).flattened_patches
            self.assertEqual(encoded_images.shape, (self.image_processor_tester.batch_size, max_patch, expected_hidden_dim))

    def test_call_numpy(self):
        if False:
            return 10
        image_processor = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)
        expected_hidden_dim = self.image_processor_tester.patch_size['height'] * self.image_processor_tester.patch_size['width'] * self.image_processor_tester.num_channels + 2
        for max_patch in self.image_processor_tester.max_patches:
            encoded_images = image_processor(image_inputs[0], return_tensors='pt', max_patches=max_patch).flattened_patches
            self.assertEqual(encoded_images.shape, (1, max_patch, expected_hidden_dim))
            encoded_images = image_processor(image_inputs, return_tensors='pt', max_patches=max_patch).flattened_patches
            self.assertEqual(encoded_images.shape, (self.image_processor_tester.batch_size, max_patch, expected_hidden_dim))

    def test_call_numpy_4_channels(self):
        if False:
            print('Hello World!')
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.image_processor_tester.num_channels = 4
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)
        expected_hidden_dim = self.image_processor_tester.patch_size['height'] * self.image_processor_tester.patch_size['width'] * self.image_processor_tester.num_channels + 2
        for max_patch in self.image_processor_tester.max_patches:
            encoded_images = image_processor(image_inputs[0], return_tensors='pt', max_patches=max_patch, input_data_format='channels_first').flattened_patches
            self.assertEqual(encoded_images.shape, (1, max_patch, expected_hidden_dim))
            encoded_images = image_processor(image_inputs, return_tensors='pt', max_patches=max_patch, input_data_format='channels_first').flattened_patches
            self.assertEqual(encoded_images.shape, (self.image_processor_tester.batch_size, max_patch, expected_hidden_dim))
        self.image_processor_tester.num_channels = 3

    def test_call_pytorch(self):
        if False:
            for i in range(10):
                print('nop')
        image_processor = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)
        expected_hidden_dim = self.image_processor_tester.patch_size['height'] * self.image_processor_tester.patch_size['width'] * self.image_processor_tester.num_channels + 2
        for max_patch in self.image_processor_tester.max_patches:
            encoded_images = image_processor(image_inputs[0], return_tensors='pt', max_patches=max_patch).flattened_patches
            self.assertEqual(encoded_images.shape, (1, max_patch, expected_hidden_dim))
            encoded_images = image_processor(image_inputs, return_tensors='pt', max_patches=max_patch).flattened_patches
            self.assertEqual(encoded_images.shape, (self.image_processor_tester.batch_size, max_patch, expected_hidden_dim))

@unittest.skipIf(not is_torch_greater_or_equal_than_1_11, reason='`Pix2StructImageProcessor` requires `torch>=1.11.0`.')
@require_torch
@require_vision
class Pix2StructImageProcessingTestFourChannels(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = Pix2StructImageProcessor if is_vision_available() else None

    def setUp(self):
        if False:
            while True:
                i = 10
        self.image_processor_tester = Pix2StructImageProcessingTester(self, num_channels=4)
        self.expected_encoded_image_num_channels = 3

    @property
    def image_processor_dict(self):
        if False:
            while True:
                i = 10
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        if False:
            return 10
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processor, 'do_normalize'))
        self.assertTrue(hasattr(image_processor, 'do_convert_rgb'))

    def test_call_pil(self):
        if False:
            i = 10
            return i + 15
        image_processor = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)
        expected_hidden_dim = self.image_processor_tester.patch_size['height'] * self.image_processor_tester.patch_size['width'] * (self.image_processor_tester.num_channels - 1) + 2
        for max_patch in self.image_processor_tester.max_patches:
            encoded_images = image_processor(image_inputs[0], return_tensors='pt', max_patches=max_patch).flattened_patches
            self.assertEqual(encoded_images.shape, (1, max_patch, expected_hidden_dim))
            encoded_images = image_processor(image_inputs, return_tensors='pt', max_patches=max_patch).flattened_patches
            self.assertEqual(encoded_images.shape, (self.image_processor_tester.batch_size, max_patch, expected_hidden_dim))

    @unittest.skip('Pix2StructImageProcessor does not support 4 channels yet')
    def test_call_numpy(self):
        if False:
            while True:
                i = 10
        return super().test_call_numpy()

    @unittest.skip('Pix2StructImageProcessor does not support 4 channels yet')
    def test_call_pytorch(self):
        if False:
            print('Hello World!')
        return super().test_call_torch()

    @unittest.skip('Pix2StructImageProcessor does treat numpy and PIL 4 channel images consistently')
    def test_call_numpy_4_channels(self):
        if False:
            while True:
                i = 10
        return super().test_call_torch()