import unittest
import numpy as np
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available
from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs
if is_torch_available():
    import torch
if is_vision_available():
    from PIL import Image
    from transformers import Swin2SRImageProcessor
    from transformers.image_transforms import get_image_size

class Swin2SRImageProcessingTester(unittest.TestCase):

    def __init__(self, parent, batch_size=7, num_channels=3, image_size=18, min_resolution=30, max_resolution=400, do_rescale=True, rescale_factor=1 / 255, do_pad=True, pad_size=8):
        if False:
            return 10
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.pad_size = pad_size

    def prepare_image_processor_dict(self):
        if False:
            return 10
        return {'do_rescale': self.do_rescale, 'rescale_factor': self.rescale_factor, 'do_pad': self.do_pad, 'pad_size': self.pad_size}

    def expected_output_image_shape(self, images):
        if False:
            for i in range(10):
                print('nop')
        img = images[0]
        if isinstance(img, Image.Image):
            (input_width, input_height) = img.size
        else:
            (input_height, input_width) = img.shape[-2:]
        pad_height = (input_height // self.pad_size + 1) * self.pad_size - input_height
        pad_width = (input_width // self.pad_size + 1) * self.pad_size - input_width
        return (self.num_channels, input_height + pad_height, input_width + pad_width)

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        if False:
            for i in range(10):
                print('nop')
        return prepare_image_inputs(batch_size=self.batch_size, num_channels=self.num_channels, min_resolution=self.min_resolution, max_resolution=self.max_resolution, equal_resolution=equal_resolution, numpify=numpify, torchify=torchify)

@require_torch
@require_vision
class Swin2SRImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = Swin2SRImageProcessor if is_vision_available() else None

    def setUp(self):
        if False:
            return 10
        self.image_processor_tester = Swin2SRImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        if False:
            return 10
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        if False:
            print('Hello World!')
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processor, 'do_rescale'))
        self.assertTrue(hasattr(image_processor, 'rescale_factor'))
        self.assertTrue(hasattr(image_processor, 'do_pad'))
        self.assertTrue(hasattr(image_processor, 'pad_size'))

    def calculate_expected_size(self, image):
        if False:
            return 10
        (old_height, old_width) = get_image_size(image)
        size = self.image_processor_tester.pad_size
        pad_height = (old_height // size + 1) * size - old_height
        pad_width = (old_width // size + 1) * size - old_width
        return (old_height + pad_height, old_width + pad_width)

    def test_call_pil(self):
        if False:
            print('Hello World!')
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)
        encoded_images = image_processing(image_inputs[0], return_tensors='pt').pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

    def test_call_numpy(self):
        if False:
            i = 10
            return i + 15
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)
        encoded_images = image_processing(image_inputs[0], return_tensors='pt').pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

    def test_call_numpy_4_channels(self):
        if False:
            while True:
                i = 10
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.image_processor_tester.num_channels = 4
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)
        encoded_images = image_processing(image_inputs[0], return_tensors='pt', input_data_format='channels_first').pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))
        self.image_processor_tester.num_channels = 3

    def test_call_pytorch(self):
        if False:
            while True:
                i = 10
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)
        encoded_images = image_processing(image_inputs[0], return_tensors='pt').pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))