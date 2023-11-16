import unittest
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available
from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs
if is_vision_available():
    from transformers import ConvNextImageProcessor

class ConvNextImageProcessingTester(unittest.TestCase):

    def __init__(self, parent, batch_size=7, num_channels=3, image_size=18, min_resolution=30, max_resolution=400, do_resize=True, size=None, crop_pct=0.875, do_normalize=True, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5]):
        if False:
            i = 10
            return i + 15
        size = size if size is not None else {'shortest_edge': 20}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.crop_pct = crop_pct
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        if False:
            i = 10
            return i + 15
        return {'image_mean': self.image_mean, 'image_std': self.image_std, 'do_normalize': self.do_normalize, 'do_resize': self.do_resize, 'size': self.size, 'crop_pct': self.crop_pct}

    def expected_output_image_shape(self, images):
        if False:
            print('Hello World!')
        return (self.num_channels, self.size['shortest_edge'], self.size['shortest_edge'])

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        if False:
            while True:
                i = 10
        return prepare_image_inputs(batch_size=self.batch_size, num_channels=self.num_channels, min_resolution=self.min_resolution, max_resolution=self.max_resolution, equal_resolution=equal_resolution, numpify=numpify, torchify=torchify)

@require_torch
@require_vision
class ConvNextImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = ConvNextImageProcessor if is_vision_available() else None

    def setUp(self):
        if False:
            print('Hello World!')
        self.image_processor_tester = ConvNextImageProcessingTester(self)

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
        self.assertTrue(hasattr(image_processing, 'do_resize'))
        self.assertTrue(hasattr(image_processing, 'size'))
        self.assertTrue(hasattr(image_processing, 'crop_pct'))
        self.assertTrue(hasattr(image_processing, 'do_normalize'))
        self.assertTrue(hasattr(image_processing, 'image_mean'))
        self.assertTrue(hasattr(image_processing, 'image_std'))

    def test_image_processor_from_dict_with_kwargs(self):
        if False:
            while True:
                i = 10
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {'shortest_edge': 20})
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42)
        self.assertEqual(image_processor.size, {'shortest_edge': 42})