import unittest
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available
from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs
if is_vision_available():
    from transformers import PoolFormerImageProcessor

class PoolFormerImageProcessingTester(unittest.TestCase):

    def __init__(self, parent, batch_size=7, num_channels=3, min_resolution=30, max_resolution=400, do_resize_and_center_crop=True, size=None, crop_pct=0.9, crop_size=None, do_normalize=True, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5]):
        if False:
            i = 10
            return i + 15
        size = size if size is not None else {'shortest_edge': 30}
        crop_size = crop_size if crop_size is not None else {'height': 30, 'width': 30}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize_and_center_crop = do_resize_and_center_crop
        self.size = size
        self.crop_pct = crop_pct
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        if False:
            i = 10
            return i + 15
        return {'size': self.size, 'do_resize_and_center_crop': self.do_resize_and_center_crop, 'crop_pct': self.crop_pct, 'crop_size': self.crop_size, 'do_normalize': self.do_normalize, 'image_mean': self.image_mean, 'image_std': self.image_std}

    def expected_output_image_shape(self, images):
        if False:
            for i in range(10):
                print('nop')
        return (self.num_channels, self.crop_size['height'], self.crop_size['width'])

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        if False:
            i = 10
            return i + 15
        return prepare_image_inputs(batch_size=self.batch_size, num_channels=self.num_channels, min_resolution=self.min_resolution, max_resolution=self.max_resolution, equal_resolution=equal_resolution, numpify=numpify, torchify=torchify)

@require_torch
@require_vision
class PoolFormerImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = PoolFormerImageProcessor if is_vision_available() else None

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.image_processor_tester = PoolFormerImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        if False:
            i = 10
            return i + 15
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        if False:
            return 10
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, 'do_resize_and_center_crop'))
        self.assertTrue(hasattr(image_processing, 'size'))
        self.assertTrue(hasattr(image_processing, 'crop_pct'))
        self.assertTrue(hasattr(image_processing, 'do_normalize'))
        self.assertTrue(hasattr(image_processing, 'image_mean'))
        self.assertTrue(hasattr(image_processing, 'image_std'))

    def test_image_processor_from_dict_with_kwargs(self):
        if False:
            return 10
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {'shortest_edge': 30})
        self.assertEqual(image_processor.crop_size, {'height': 30, 'width': 30})
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42, crop_size=84)
        self.assertEqual(image_processor.size, {'shortest_edge': 42})
        self.assertEqual(image_processor.crop_size, {'height': 84, 'width': 84})