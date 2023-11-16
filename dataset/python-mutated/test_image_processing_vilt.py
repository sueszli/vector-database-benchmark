import unittest
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available
from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs
if is_vision_available():
    from PIL import Image
    from transformers import ViltImageProcessor

class ViltImageProcessingTester(unittest.TestCase):

    def __init__(self, parent, batch_size=7, num_channels=3, image_size=18, min_resolution=30, max_resolution=400, do_resize=True, size=None, size_divisor=2, do_normalize=True, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5]):
        if False:
            print('Hello World!')
        size = size if size is not None else {'shortest_edge': 30}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.size_divisor = size_divisor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        if False:
            print('Hello World!')
        return {'image_mean': self.image_mean, 'image_std': self.image_std, 'do_normalize': self.do_normalize, 'do_resize': self.do_resize, 'size': self.size, 'size_divisor': self.size_divisor}

    def get_expected_values(self, image_inputs, batched=False):
        if False:
            print('Hello World!')
        '\n        This function computes the expected height and width when providing images to ViltImageProcessor,\n        assuming do_resize is set to True with a scalar size and size_divisor.\n        '
        if not batched:
            size = self.size['shortest_edge']
            image = image_inputs[0]
            if isinstance(image, Image.Image):
                (w, h) = image.size
            else:
                (h, w) = (image.shape[1], image.shape[2])
            scale = size / min(w, h)
            if h < w:
                (newh, neww) = (size, scale * w)
            else:
                (newh, neww) = (scale * h, size)
            max_size = int(1333 / 800 * size)
            if max(newh, neww) > max_size:
                scale = max_size / max(newh, neww)
                newh = newh * scale
                neww = neww * scale
            (newh, neww) = (int(newh + 0.5), int(neww + 0.5))
            (expected_height, expected_width) = (newh // self.size_divisor * self.size_divisor, neww // self.size_divisor * self.size_divisor)
        else:
            expected_values = []
            for image in image_inputs:
                (expected_height, expected_width) = self.get_expected_values([image])
                expected_values.append((expected_height, expected_width))
            expected_height = max(expected_values, key=lambda item: item[0])[0]
            expected_width = max(expected_values, key=lambda item: item[1])[1]
        return (expected_height, expected_width)

    def expected_output_image_shape(self, images):
        if False:
            print('Hello World!')
        (height, width) = self.get_expected_values(images, batched=True)
        return (self.num_channels, height, width)

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        if False:
            while True:
                i = 10
        return prepare_image_inputs(batch_size=self.batch_size, num_channels=self.num_channels, min_resolution=self.min_resolution, max_resolution=self.max_resolution, equal_resolution=equal_resolution, numpify=numpify, torchify=torchify)

@require_torch
@require_vision
class ViltImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = ViltImageProcessor if is_vision_available() else None

    def setUp(self):
        if False:
            return 10
        self.image_processor_tester = ViltImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        if False:
            return 10
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        if False:
            for i in range(10):
                print('nop')
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, 'image_mean'))
        self.assertTrue(hasattr(image_processing, 'image_std'))
        self.assertTrue(hasattr(image_processing, 'do_normalize'))
        self.assertTrue(hasattr(image_processing, 'do_resize'))
        self.assertTrue(hasattr(image_processing, 'size'))
        self.assertTrue(hasattr(image_processing, 'size_divisor'))

    def test_image_processor_from_dict_with_kwargs(self):
        if False:
            return 10
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict)
        self.assertEqual(image_processor.size, {'shortest_edge': 30})
        image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42)
        self.assertEqual(image_processor.size, {'shortest_edge': 42})