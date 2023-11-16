import unittest
import numpy as np
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available
from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs
if is_torch_available():
    import torch
if is_vision_available():
    from PIL import Image
    from transformers import VitMatteImageProcessor

class VitMatteImageProcessingTester(unittest.TestCase):

    def __init__(self, parent, batch_size=7, num_channels=3, image_size=18, min_resolution=30, max_resolution=400, do_rescale=True, rescale_factor=0.5, do_pad=True, size_divisibility=10, do_normalize=True, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5]):
        if False:
            print('Hello World!')
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.size_divisibility = size_divisibility
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        if False:
            return 10
        return {'image_mean': self.image_mean, 'image_std': self.image_std, 'do_normalize': self.do_normalize, 'do_rescale': self.do_rescale, 'rescale_factor': self.rescale_factor, 'do_pad': self.do_pad, 'size_divisibility': self.size_divisibility}

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        if False:
            while True:
                i = 10
        return prepare_image_inputs(batch_size=self.batch_size, num_channels=self.num_channels, min_resolution=self.min_resolution, max_resolution=self.max_resolution, equal_resolution=equal_resolution, numpify=numpify, torchify=torchify)

@require_torch
@require_vision
class VitMatteImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = VitMatteImageProcessor if is_vision_available() else None

    def setUp(self):
        if False:
            while True:
                i = 10
        self.image_processor_tester = VitMatteImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        if False:
            print('Hello World!')
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        if False:
            while True:
                i = 10
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, 'image_mean'))
        self.assertTrue(hasattr(image_processing, 'image_std'))
        self.assertTrue(hasattr(image_processing, 'do_normalize'))
        self.assertTrue(hasattr(image_processing, 'do_rescale'))
        self.assertTrue(hasattr(image_processing, 'rescale_factor'))
        self.assertTrue(hasattr(image_processing, 'do_pad'))
        self.assertTrue(hasattr(image_processing, 'size_divisibility'))

    def test_call_numpy(self):
        if False:
            while True:
                i = 10
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.shape[:2])
        encoded_images = image_processing(images=image, trimaps=trimap, return_tensors='pt').pixel_values
        self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisibility == 0)
        self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisibility == 0)

    def test_call_pytorch(self):
        if False:
            i = 10
            return i + 15
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.shape[:2])
        encoded_images = image_processing(images=image, trimaps=trimap, return_tensors='pt').pixel_values
        self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisibility == 0)
        self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisibility == 0)

    def test_call_pil(self):
        if False:
            while True:
                i = 10
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.size[::-1])
        encoded_images = image_processing(images=image, trimaps=trimap, return_tensors='pt').pixel_values
        self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisibility == 0)
        self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisibility == 0)

    def test_call_numpy_4_channels(self):
        if False:
            while True:
                i = 10
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.image_processor_tester.num_channels = 4
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        image = image_inputs[0]
        trimap = np.random.randint(0, 3, size=image.shape[:2])
        encoded_images = image_processor(images=image, trimaps=trimap, input_data_format='channels_first', image_mean=0, image_std=1, return_tensors='pt').pixel_values
        self.assertTrue(encoded_images.shape[-1] % self.image_processor_tester.size_divisibility == 0)
        self.assertTrue(encoded_images.shape[-2] % self.image_processor_tester.size_divisibility == 0)

    def test_padding(self):
        if False:
            while True:
                i = 10
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image = np.random.randn(3, 249, 491)
        images = image_processing.pad_image(image)
        assert images.shape == (3, 256, 512)