""" Testing suite for the TVLT image processor. """
import unittest
import numpy as np
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available
from ...test_image_processing_common import ImageProcessingTestMixin
if is_torch_available():
    import torch
if is_vision_available():
    from PIL import Image
    from transformers import TvltImageProcessor

def prepare_video(image_processor_tester, width=10, height=10, numpify=False, torchify=False):
    if False:
        print('Hello World!')
    'This function prepares a video as a list of PIL images/NumPy arrays/PyTorch tensors.'
    video = []
    for i in range(image_processor_tester.num_frames):
        video.append(np.random.randint(255, size=(image_processor_tester.num_channels, width, height), dtype=np.uint8))
    if not numpify and (not torchify):
        video = [Image.fromarray(np.moveaxis(frame, 0, -1)) for frame in video]
    if torchify:
        video = [torch.from_numpy(frame) for frame in video]
    return video

def prepare_video_inputs(image_processor_tester, equal_resolution=False, numpify=False, torchify=False):
    if False:
        print('Hello World!')
    'This function prepares a batch of videos: a list of list of PIL images, or a list of list of numpy arrays if\n    one specifies numpify=True, or a list of list of PyTorch tensors if one specifies torchify=True.\n    One can specify whether the videos are of the same resolution or not.\n    '
    assert not (numpify and torchify), 'You cannot specify both numpy and PyTorch tensors at the same time'
    video_inputs = []
    for i in range(image_processor_tester.batch_size):
        if equal_resolution:
            width = height = image_processor_tester.max_resolution
        else:
            (width, height) = np.random.choice(np.arange(image_processor_tester.min_resolution, image_processor_tester.max_resolution), 2)
            video = prepare_video(image_processor_tester=image_processor_tester, width=width, height=height, numpify=numpify, torchify=torchify)
        video_inputs.append(video)
    return video_inputs

class TvltImageProcessorTester(unittest.TestCase):

    def __init__(self, parent, batch_size=7, num_channels=3, num_frames=4, image_size=18, min_resolution=30, max_resolution=400, do_resize=True, size=None, do_normalize=True, image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5], do_center_crop=True, crop_size=None):
        if False:
            print('Hello World!')
        size = size if size is not None else {'shortest_edge': 18}
        crop_size = crop_size if crop_size is not None else {'height': 18, 'width': 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size

    def prepare_image_processor_dict(self):
        if False:
            print('Hello World!')
        return {'image_mean': self.image_mean, 'image_std': self.image_std, 'do_normalize': self.do_normalize, 'do_resize': self.do_resize, 'size': self.size, 'do_center_crop': self.do_center_crop, 'crop_size': self.crop_size}

@require_torch
@require_vision
class TvltImageProcessorTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = TvltImageProcessor if is_vision_available() else None

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.image_processor_tester = TvltImageProcessorTester(self)

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
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processor, 'image_mean'))
        self.assertTrue(hasattr(image_processor, 'image_std'))
        self.assertTrue(hasattr(image_processor, 'do_normalize'))
        self.assertTrue(hasattr(image_processor, 'do_resize'))
        self.assertTrue(hasattr(image_processor, 'do_center_crop'))
        self.assertTrue(hasattr(image_processor, 'size'))

    def test_call_pil(self):
        if False:
            print('Hello World!')
        image_processor = self.image_processing_class(**self.image_processor_dict)
        video_inputs = prepare_video_inputs(self.image_processor_tester, equal_resolution=False)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], Image.Image)
        encoded_videos = image_processor(video_inputs[0], return_tensors='pt').pixel_values
        self.assertEqual(encoded_videos.shape, (1, self.image_processor_tester.num_frames, self.image_processor_tester.num_channels, self.image_processor_tester.crop_size['height'], self.image_processor_tester.crop_size['width']))
        encoded_videos = image_processor(video_inputs, return_tensors='pt').pixel_values
        self.assertEqual(encoded_videos.shape, (self.image_processor_tester.batch_size, self.image_processor_tester.num_frames, self.image_processor_tester.num_channels, self.image_processor_tester.crop_size['height'], self.image_processor_tester.crop_size['width']))

    def test_call_numpy(self):
        if False:
            i = 10
            return i + 15
        image_processor = self.image_processing_class(**self.image_processor_dict)
        video_inputs = prepare_video_inputs(self.image_processor_tester, equal_resolution=False, numpify=True)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], np.ndarray)
        encoded_videos = image_processor(video_inputs[0], return_tensors='pt').pixel_values
        self.assertEqual(encoded_videos.shape, (1, self.image_processor_tester.num_frames, self.image_processor_tester.num_channels, self.image_processor_tester.crop_size['height'], self.image_processor_tester.crop_size['width']))
        encoded_videos = image_processor(video_inputs, return_tensors='pt').pixel_values
        self.assertEqual(encoded_videos.shape, (self.image_processor_tester.batch_size, self.image_processor_tester.num_frames, self.image_processor_tester.num_channels, self.image_processor_tester.crop_size['height'], self.image_processor_tester.crop_size['width']))

    def test_call_numpy_4_channels(self):
        if False:
            print('Hello World!')
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.image_processor_tester.num_channels = 4
        video_inputs = prepare_video_inputs(self.image_processor_tester, equal_resolution=False, numpify=True)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], np.ndarray)
        encoded_videos = image_processor(video_inputs[0], return_tensors='pt', input_data_format='channels_first', image_mean=0, image_std=1).pixel_values
        self.assertEqual(encoded_videos.shape, (1, self.image_processor_tester.num_frames, self.image_processor_tester.num_channels, self.image_processor_tester.crop_size['height'], self.image_processor_tester.crop_size['width']))
        encoded_videos = image_processor(video_inputs, return_tensors='pt', input_data_format='channels_first', image_mean=0, image_std=1).pixel_values
        self.assertEqual(encoded_videos.shape, (self.image_processor_tester.batch_size, self.image_processor_tester.num_frames, self.image_processor_tester.num_channels, self.image_processor_tester.crop_size['height'], self.image_processor_tester.crop_size['width']))
        self.image_processor_tester.num_channels = 3

    def test_call_pytorch(self):
        if False:
            i = 10
            return i + 15
        image_processor = self.image_processing_class(**self.image_processor_dict)
        video_inputs = prepare_video_inputs(self.image_processor_tester, equal_resolution=False, torchify=True)
        for video in video_inputs:
            self.assertIsInstance(video, list)
            self.assertIsInstance(video[0], torch.Tensor)
        encoded_videos = image_processor(video_inputs[0], return_tensors='pt').pixel_values
        self.assertEqual(encoded_videos.shape, (1, self.image_processor_tester.num_frames, self.image_processor_tester.num_channels, self.image_processor_tester.crop_size['height'], self.image_processor_tester.crop_size['width']))
        encoded_videos = image_processor(video_inputs, return_tensors='pt').pixel_values
        self.assertEqual(encoded_videos.shape, (self.image_processor_tester.batch_size, self.image_processor_tester.num_frames, self.image_processor_tester.num_channels, self.image_processor_tester.crop_size['height'], self.image_processor_tester.crop_size['width']))