import json
import os
import tempfile
from transformers.testing_utils import check_json_file_has_correct_format, require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available
if is_torch_available():
    import numpy as np
    import torch
if is_vision_available():
    from PIL import Image

def prepare_image_inputs(batch_size, min_resolution, max_resolution, num_channels, size_divisor=None, equal_resolution=False, numpify=False, torchify=False):
    if False:
        while True:
            i = 10
    'This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,\n    or a list of PyTorch tensors if one specifies torchify=True.\n\n    One can specify whether the images are of the same resolution or not.\n    '
    assert not (numpify and torchify), 'You cannot specify both numpy and PyTorch tensors at the same time'
    image_inputs = []
    for i in range(batch_size):
        if equal_resolution:
            width = height = max_resolution
        else:
            if size_divisor is not None:
                min_resolution = max(size_divisor, min_resolution)
            (width, height) = np.random.choice(np.arange(min_resolution, max_resolution), 2)
        image_inputs.append(np.random.randint(255, size=(num_channels, width, height), dtype=np.uint8))
    if not numpify and (not torchify):
        image_inputs = [Image.fromarray(np.moveaxis(image, 0, -1)) for image in image_inputs]
    if torchify:
        image_inputs = [torch.from_numpy(image) for image in image_inputs]
    return image_inputs

def prepare_video(num_frames, num_channels, width=10, height=10, numpify=False, torchify=False):
    if False:
        i = 10
        return i + 15
    'This function prepares a video as a list of PIL images/NumPy arrays/PyTorch tensors.'
    video = []
    for i in range(num_frames):
        video.append(np.random.randint(255, size=(num_channels, width, height), dtype=np.uint8))
    if not numpify and (not torchify):
        video = [Image.fromarray(np.moveaxis(frame, 0, -1)) for frame in video]
    if torchify:
        video = [torch.from_numpy(frame) for frame in video]
    return video

def prepare_video_inputs(batch_size, num_frames, num_channels, min_resolution, max_resolution, equal_resolution=False, numpify=False, torchify=False):
    if False:
        i = 10
        return i + 15
    'This function prepares a batch of videos: a list of list of PIL images, or a list of list of numpy arrays if\n    one specifies numpify=True, or a list of list of PyTorch tensors if one specifies torchify=True.\n\n    One can specify whether the videos are of the same resolution or not.\n    '
    assert not (numpify and torchify), 'You cannot specify both numpy and PyTorch tensors at the same time'
    video_inputs = []
    for i in range(batch_size):
        if equal_resolution:
            width = height = max_resolution
        else:
            (width, height) = np.random.choice(np.arange(min_resolution, max_resolution), 2)
            video = prepare_video(num_frames=num_frames, num_channels=num_channels, width=width, height=height, numpify=numpify, torchify=torchify)
        video_inputs.append(video)
    return video_inputs

class ImageProcessingTestMixin:
    test_cast_dtype = None

    def test_image_processor_to_json_string(self):
        if False:
            for i in range(10):
                print('nop')
        image_processor = self.image_processing_class(**self.image_processor_dict)
        obj = json.loads(image_processor.to_json_string())
        for (key, value) in self.image_processor_dict.items():
            self.assertEqual(obj[key], value)

    def test_image_processor_to_json_file(self):
        if False:
            i = 10
            return i + 15
        image_processor_first = self.image_processing_class(**self.image_processor_dict)
        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, 'image_processor.json')
            image_processor_first.to_json_file(json_file_path)
            image_processor_second = self.image_processing_class.from_json_file(json_file_path)
        self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

    def test_image_processor_from_and_save_pretrained(self):
        if False:
            print('Hello World!')
        image_processor_first = self.image_processing_class(**self.image_processor_dict)
        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = image_processor_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            image_processor_second = self.image_processing_class.from_pretrained(tmpdirname)
        self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

    def test_init_without_params(self):
        if False:
            print('Hello World!')
        image_processor = self.image_processing_class()
        self.assertIsNotNone(image_processor)

    @require_torch
    @require_vision
    def test_cast_dtype_device(self):
        if False:
            while True:
                i = 10
        if self.test_cast_dtype is not None:
            image_processor = self.image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
            encoding = image_processor(image_inputs, return_tensors='pt')
            self.assertEqual(encoding.pixel_values.device, torch.device('cpu'))
            self.assertEqual(encoding.pixel_values.dtype, torch.float32)
            encoding = image_processor(image_inputs, return_tensors='pt').to(torch.float16)
            self.assertEqual(encoding.pixel_values.device, torch.device('cpu'))
            self.assertEqual(encoding.pixel_values.dtype, torch.float16)
            encoding = image_processor(image_inputs, return_tensors='pt').to('cpu', torch.bfloat16)
            self.assertEqual(encoding.pixel_values.device, torch.device('cpu'))
            self.assertEqual(encoding.pixel_values.dtype, torch.bfloat16)
            with self.assertRaises(TypeError):
                _ = image_processor(image_inputs, return_tensors='pt').to(torch.bfloat16, 'cpu')
            encoding = image_processor(image_inputs, return_tensors='pt')
            encoding.update({'input_ids': torch.LongTensor([[1, 2, 3], [4, 5, 6]])})
            encoding = encoding.to(torch.float16)
            self.assertEqual(encoding.pixel_values.device, torch.device('cpu'))
            self.assertEqual(encoding.pixel_values.dtype, torch.float16)
            self.assertEqual(encoding.input_ids.dtype, torch.long)

    def test_call_pil(self):
        if False:
            while True:
                i = 10
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)
        encoded_images = image_processing(image_inputs[0], return_tensors='pt').pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))
        encoded_images = image_processing(image_inputs, return_tensors='pt').pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        self.assertEqual(tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape))

    def test_call_numpy(self):
        if False:
            return 10
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)
        encoded_images = image_processing(image_inputs[0], return_tensors='pt').pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))
        encoded_images = image_processing(image_inputs, return_tensors='pt').pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        self.assertEqual(tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape))

    def test_call_pytorch(self):
        if False:
            i = 10
            return i + 15
        image_processing = self.image_processing_class(**self.image_processor_dict)
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)
        encoded_images = image_processing(image_inputs[0], return_tensors='pt').pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        encoded_images = image_processing(image_inputs, return_tensors='pt').pixel_values
        self.assertEqual(tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape))

    def test_call_numpy_4_channels(self):
        if False:
            while True:
                i = 10
        image_processor = self.image_processing_class(**self.image_processor_dict)
        self.image_processor_tester.num_channels = 4
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        encoded_images = image_processor(image_inputs[0], return_tensors='pt', input_data_format='channels_first', image_mean=0, image_std=1).pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))
        encoded_images = image_processor(image_inputs, return_tensors='pt', input_data_format='channels_first', image_mean=0, image_std=1).pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        self.assertEqual(tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape))