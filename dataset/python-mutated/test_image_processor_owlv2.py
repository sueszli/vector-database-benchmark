import unittest
from transformers.testing_utils import require_torch, require_vision, slow
from transformers.utils import is_vision_available
from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs
if is_vision_available():
    from PIL import Image
    from transformers import Owlv2ImageProcessor

class Owlv2ImageProcessingTester(unittest.TestCase):

    def __init__(self, parent, batch_size=7, num_channels=3, image_size=18, min_resolution=30, max_resolution=400, do_resize=True, size=None, do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073], image_std=[0.26862954, 0.26130258, 0.27577711], do_convert_rgb=True):
        if False:
            for i in range(10):
                print('nop')
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size if size is not None else {'height': 18, 'width': 18}
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        if False:
            i = 10
            return i + 15
        return {'do_resize': self.do_resize, 'size': self.size, 'do_normalize': self.do_normalize, 'image_mean': self.image_mean, 'image_std': self.image_std}

    def expected_output_image_shape(self, images):
        if False:
            i = 10
            return i + 15
        return (self.num_channels, self.size['height'], self.size['width'])

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        if False:
            for i in range(10):
                print('nop')
        return prepare_image_inputs(batch_size=self.batch_size, num_channels=self.num_channels, min_resolution=self.min_resolution, max_resolution=self.max_resolution, equal_resolution=equal_resolution, numpify=numpify, torchify=torchify)

@require_torch
@require_vision
class Owlv2ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = Owlv2ImageProcessor if is_vision_available() else None

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.image_processor_tester = Owlv2ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        if False:
            for i in range(10):
                print('nop')
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        if False:
            return 10
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, 'do_resize'))
        self.assertTrue(hasattr(image_processing, 'size'))
        self.assertTrue(hasattr(image_processing, 'do_normalize'))
        self.assertTrue(hasattr(image_processing, 'image_mean'))
        self.assertTrue(hasattr(image_processing, 'image_std'))

    def test_image_processor_from_dict_with_kwargs(self):
        if False:
            i = 10
            return i + 15
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {'height': 18, 'width': 18})
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size={'height': 42, 'width': 42})
        self.assertEqual(image_processor.size, {'height': 42, 'width': 42})

    @slow
    def test_image_processor_integration_test(self):
        if False:
            for i in range(10):
                print('nop')
        processor = Owlv2ImageProcessor()
        image = Image.open('./tests/fixtures/tests_samples/COCO/000000039769.png')
        pixel_values = processor(image, return_tensors='pt').pixel_values
        mean_value = round(pixel_values.mean().item(), 4)
        self.assertEqual(mean_value, 0.2353)

    @unittest.skip("OWLv2 doesn't treat 4 channel PIL and numpy consistently yet")
    def test_call_numpy_4_channels(self):
        if False:
            i = 10
            return i + 15
        pass