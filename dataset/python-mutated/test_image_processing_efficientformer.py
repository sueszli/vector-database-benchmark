import unittest
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available
from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs
if is_vision_available():
    from transformers import ViTImageProcessor

class EfficientFormerImageProcessorTester(unittest.TestCase):

    def __init__(self, parent, batch_size=13, num_channels=3, image_size=224, min_resolution=30, max_resolution=400, do_resize=True, size=None, do_normalize=True, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5]):
        if False:
            i = 10
            return i + 15
        size = size if size is not None else {'height': 18, 'width': 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        if False:
            for i in range(10):
                print('nop')
        return {'image_mean': self.image_mean, 'image_std': self.image_std, 'do_normalize': self.do_normalize, 'do_resize': self.do_resize, 'size': self.size}

    def expected_output_image_shape(self, images):
        if False:
            i = 10
            return i + 15
        return (self.num_channels, self.size['height'], self.size['width'])

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        if False:
            return 10
        return prepare_image_inputs(batch_size=self.batch_size, num_channels=self.num_channels, min_resolution=self.min_resolution, max_resolution=self.max_resolution, equal_resolution=equal_resolution, numpify=numpify, torchify=torchify)

@require_torch
@require_vision
class EfficientFormerImageProcessorTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = ViTImageProcessor if is_vision_available() else None

    def setUp(self):
        if False:
            return 10
        self.image_processor_tester = EfficientFormerImageProcessorTester(self)

    @property
    def image_processor_dict(self):
        if False:
            print('Hello World!')
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_proc_properties(self):
        if False:
            return 10
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processor, 'image_mean'))
        self.assertTrue(hasattr(image_processor, 'image_std'))
        self.assertTrue(hasattr(image_processor, 'do_normalize'))
        self.assertTrue(hasattr(image_processor, 'do_resize'))
        self.assertTrue(hasattr(image_processor, 'size'))