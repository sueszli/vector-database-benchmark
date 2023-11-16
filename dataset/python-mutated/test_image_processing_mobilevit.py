import unittest
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available
from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs
if is_vision_available():
    from transformers import MobileViTImageProcessor

class MobileViTImageProcessingTester(unittest.TestCase):

    def __init__(self, parent, batch_size=7, num_channels=3, image_size=18, min_resolution=30, max_resolution=400, do_resize=True, size=None, do_center_crop=True, crop_size=None, do_flip_channel_order=True):
        if False:
            i = 10
            return i + 15
        size = size if size is not None else {'shortest_edge': 20}
        crop_size = crop_size if crop_size is not None else {'height': 18, 'width': 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_flip_channel_order = do_flip_channel_order

    def prepare_image_processor_dict(self):
        if False:
            for i in range(10):
                print('nop')
        return {'do_resize': self.do_resize, 'size': self.size, 'do_center_crop': self.do_center_crop, 'crop_size': self.crop_size, 'do_flip_channel_order': self.do_flip_channel_order}

    def expected_output_image_shape(self, images):
        if False:
            return 10
        return (self.num_channels, self.crop_size['height'], self.crop_size['width'])

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        if False:
            while True:
                i = 10
        return prepare_image_inputs(batch_size=self.batch_size, num_channels=self.num_channels, min_resolution=self.min_resolution, max_resolution=self.max_resolution, equal_resolution=equal_resolution, numpify=numpify, torchify=torchify)

@require_torch
@require_vision
class MobileViTImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = MobileViTImageProcessor if is_vision_available() else None

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.image_processor_tester = MobileViTImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        if False:
            while True:
                i = 10
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        if False:
            i = 10
            return i + 15
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, 'do_resize'))
        self.assertTrue(hasattr(image_processing, 'size'))
        self.assertTrue(hasattr(image_processing, 'do_center_crop'))
        self.assertTrue(hasattr(image_processing, 'center_crop'))
        self.assertTrue(hasattr(image_processing, 'do_flip_channel_order'))

    def test_image_processor_from_dict_with_kwargs(self):
        if False:
            return 10
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {'shortest_edge': 20})
        self.assertEqual(image_processor.crop_size, {'height': 18, 'width': 18})
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42, crop_size=84)
        self.assertEqual(image_processor.size, {'shortest_edge': 42})
        self.assertEqual(image_processor.crop_size, {'height': 84, 'width': 84})