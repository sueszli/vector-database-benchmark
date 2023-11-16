import unittest
import numpy as np
from parameterized import parameterized
from transformers.testing_utils import require_flax, require_tf, require_torch, require_vision
from transformers.utils.import_utils import is_flax_available, is_tf_available, is_torch_available, is_vision_available
if is_torch_available():
    import torch
if is_tf_available():
    import tensorflow as tf
if is_flax_available():
    import jax
if is_vision_available():
    import PIL.Image
    from transformers.image_transforms import center_crop, center_to_corners_format, convert_to_rgb, corners_to_center_format, flip_channel_order, get_resize_output_image_size, id_to_rgb, normalize, pad, resize, rgb_to_id, to_channel_dimension_format, to_pil_image

def get_random_image(height, width, num_channels=3, channels_first=True):
    if False:
        print('Hello World!')
    shape = (num_channels, height, width) if channels_first else (height, width, num_channels)
    random_array = np.random.randint(0, 256, shape, dtype=np.uint8)
    return random_array

@require_vision
class ImageTransformsTester(unittest.TestCase):

    @parameterized.expand([('numpy_float_channels_first', (3, 4, 5), np.float32), ('numpy_float_channels_last', (4, 5, 3), np.float32), ('numpy_float_channels_first', (3, 4, 5), np.float64), ('numpy_float_channels_last', (4, 5, 3), np.float64), ('numpy_int_channels_first', (3, 4, 5), np.int32), ('numpy_uint_channels_first', (3, 4, 5), np.uint8)])
    @require_vision
    def test_to_pil_image(self, name, image_shape, dtype):
        if False:
            print('Hello World!')
        image = np.random.randint(0, 256, image_shape).astype(dtype)
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))
        self.assertTrue(np.abs(np.asarray(pil_image)).sum() > 0)

    @parameterized.expand([('numpy_float_channels_first', (3, 4, 5), np.float32), ('numpy_float_channels_first', (3, 4, 5), np.float64), ('numpy_float_channels_last', (4, 5, 3), np.float32), ('numpy_float_channels_last', (4, 5, 3), np.float64)])
    @require_vision
    def test_to_pil_image_from_float(self, name, image_shape, dtype):
        if False:
            return 10
        image = np.random.rand(*image_shape).astype(dtype)
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))
        self.assertTrue(np.abs(np.asarray(pil_image)).sum() > 0)
        image = np.random.randn(*image_shape).astype(dtype)
        with self.assertRaises(ValueError):
            to_pil_image(image)

    @require_vision
    def test_to_pil_image_from_mask(self):
        if False:
            print('Hello World!')
        image = np.random.randint(0, 2, (3, 4, 5)).astype(np.uint8)
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))
        np_img = np.asarray(pil_image)
        self.assertTrue(np_img.min() == 0)
        self.assertTrue(np_img.max() == 1)
        image = np.random.randint(0, 2, (3, 4, 5)).astype(np.float32)
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))
        np_img = np.asarray(pil_image)
        self.assertTrue(np_img.min() == 0)
        self.assertTrue(np_img.max() == 1)

    @require_tf
    def test_to_pil_image_from_tensorflow(self):
        if False:
            print('Hello World!')
        image = tf.random.uniform((3, 4, 5))
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))
        image = tf.random.uniform((4, 5, 3))
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))

    @require_torch
    def test_to_pil_image_from_torch(self):
        if False:
            for i in range(10):
                print('nop')
        image = torch.rand((3, 4, 5))
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))
        image = torch.rand((4, 5, 3))
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))

    @require_flax
    def test_to_pil_image_from_jax(self):
        if False:
            i = 10
            return i + 15
        key = jax.random.PRNGKey(0)
        image = jax.random.uniform(key, (3, 4, 5))
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))
        image = jax.random.uniform(key, (4, 5, 3))
        pil_image = to_pil_image(image)
        self.assertIsInstance(pil_image, PIL.Image.Image)
        self.assertEqual(pil_image.size, (5, 4))

    def test_to_channel_dimension_format(self):
        if False:
            print('Hello World!')
        image = np.random.rand(3, 4, 5)
        image = to_channel_dimension_format(image, 'channels_first')
        self.assertEqual(image.shape, (3, 4, 5))
        image = np.random.rand(4, 5, 3)
        image = to_channel_dimension_format(image, 'channels_last')
        self.assertEqual(image.shape, (4, 5, 3))
        image = np.random.rand(3, 4, 5)
        image = to_channel_dimension_format(image, 'channels_last')
        self.assertEqual(image.shape, (4, 5, 3))
        image = np.random.rand(4, 5, 3)
        image = to_channel_dimension_format(image, 'channels_first')
        self.assertEqual(image.shape, (3, 4, 5))
        image = np.random.rand(4, 5, 6)
        image = to_channel_dimension_format(image, 'channels_first', input_channel_dim='channels_last')
        self.assertEqual(image.shape, (6, 4, 5))

    def test_get_resize_output_image_size(self):
        if False:
            for i in range(10):
                print('nop')
        image = np.random.randint(0, 256, (3, 224, 224))
        self.assertEqual(get_resize_output_image_size(image, 10), (10, 10))
        self.assertEqual(get_resize_output_image_size(image, [10]), (10, 10))
        self.assertEqual(get_resize_output_image_size(image, (10,)), (10, 10))
        self.assertEqual(get_resize_output_image_size(image, (10, 20)), (10, 20))
        self.assertEqual(get_resize_output_image_size(image, [10, 20]), (10, 20))
        self.assertEqual(get_resize_output_image_size(image, (10, 20), default_to_square=True), (10, 20))
        self.assertEqual(get_resize_output_image_size(image, (10, 20), max_size=5), (10, 20))
        image = np.random.randint(0, 256, (3, 50, 40))
        self.assertEqual(get_resize_output_image_size(image, 20, default_to_square=False), (25, 20))
        image = np.random.randint(0, 256, (3, 40, 50))
        self.assertEqual(get_resize_output_image_size(image, 20, default_to_square=False), (20, 25))
        image = np.random.randint(0, 256, (3, 50, 40))
        self.assertEqual(get_resize_output_image_size(image, 20, default_to_square=False, max_size=22), (22, 17))
        image = np.random.randint(0, 256, (4, 50, 40))
        self.assertEqual(get_resize_output_image_size(image, 20, default_to_square=False, input_data_format='channels_first'), (25, 20))
        image = np.random.randint(0, 256, (3, 18, 97))
        resized_image = resize(image, (3, 20))
        self.assertEqual(resized_image.shape, (3, 3, 20))
        image = np.random.randint(0, 256, (18, 97, 3))
        resized_image = resize(image, (3, 20))
        self.assertEqual(resized_image.shape, (3, 20, 3))
        image = np.random.randint(0, 256, (3, 18, 97))
        resized_image = resize(image, (3, 20), data_format='channels_last')
        self.assertEqual(resized_image.shape, (3, 20, 3))
        image = np.random.randint(0, 256, (18, 97, 3))
        resized_image = resize(image, (3, 20), data_format='channels_first')
        self.assertEqual(resized_image.shape, (3, 3, 20))

    def test_resize(self):
        if False:
            for i in range(10):
                print('nop')
        image = np.random.randint(0, 256, (3, 224, 224))
        resized_image = resize(image, (30, 40))
        self.assertIsInstance(resized_image, np.ndarray)
        self.assertEqual(resized_image.shape, (3, 30, 40))
        resized_image = resize(image, (30, 40), data_format='channels_last')
        self.assertIsInstance(resized_image, np.ndarray)
        self.assertEqual(resized_image.shape, (30, 40, 3))
        resized_image = resize(image, (30, 40), return_numpy=False)
        self.assertIsInstance(resized_image, PIL.Image.Image)
        self.assertEqual(resized_image.size, (40, 30))
        image = np.random.rand(3, 224, 224)
        resized_image = resize(image, (30, 40))
        self.assertIsInstance(resized_image, np.ndarray)
        self.assertEqual(resized_image.shape, (3, 30, 40))
        self.assertTrue(np.all(resized_image >= 0))
        self.assertTrue(np.all(resized_image <= 1))
        image = np.random.randint(0, 256, (4, 224, 224))
        resized_image = resize(image, (30, 40), input_data_format='channels_first')
        self.assertIsInstance(resized_image, np.ndarray)
        self.assertEqual(resized_image.shape, (4, 30, 40))

    def test_normalize(self):
        if False:
            for i in range(10):
                print('nop')
        image = np.random.randint(0, 256, (224, 224, 3)) / 255
        with self.assertRaises(ValueError):
            normalize(5, 5, 5)
        with self.assertRaises(ValueError):
            normalize(image, mean=(0.5, 0.6), std=1)
        with self.assertRaises(ValueError):
            normalize(image, mean=1, std=(0.5, 0.6))
        mean = (0.5, 0.6, 0.7)
        std = (0.1, 0.2, 0.3)
        expected_image = ((image - mean) / std).transpose((2, 0, 1))
        normalized_image = normalize(image, mean=mean, std=std, data_format='channels_first')
        self.assertIsInstance(normalized_image, np.ndarray)
        self.assertEqual(normalized_image.shape, (3, 224, 224))
        self.assertTrue(np.allclose(normalized_image, expected_image, atol=1e-06))
        image = np.random.randint(0, 256, (224, 224, 4)) / 255
        mean = (0.5, 0.6, 0.7, 0.8)
        std = (0.1, 0.2, 0.3, 0.4)
        expected_image = (image - mean) / std
        self.assertTrue(np.allclose(normalize(image, mean=mean, std=std, input_data_format='channels_last'), expected_image, atol=1e-06))
        image = np.random.randint(0, 256, (224, 224, 3)).astype(np.float32) / 255
        mean = (0.5, 0.6, 0.7)
        std = (0.1, 0.2, 0.3)
        expected_image = ((image - mean) / std).astype(np.float32)
        normalized_image = normalize(image, mean=mean, std=std)
        self.assertEqual(normalized_image.dtype, np.float32)
        self.assertTrue(np.allclose(normalized_image, expected_image, atol=1e-06))
        image = np.random.randint(0, 256, (224, 224, 3)).astype(np.float16) / 255
        mean = (0.5, 0.6, 0.7)
        std = (0.1, 0.2, 0.3)
        cast_mean = np.array(mean, dtype=np.float16)
        cast_std = np.array(std, dtype=np.float16)
        expected_image = (image - cast_mean) / cast_std
        normalized_image = normalize(image, mean=mean, std=std)
        self.assertEqual(normalized_image.dtype, np.float16)
        self.assertTrue(np.allclose(normalized_image, expected_image, atol=1e-06))
        image = np.random.randint(0, 2, (224, 224, 3), dtype=np.uint8)
        mean = (0.5, 0.6, 0.7)
        std = (0.1, 0.2, 0.3)
        expected_image = (image.astype(np.float32) - mean) / std
        normalized_image = normalize(image, mean=mean, std=std)
        self.assertEqual(normalized_image.dtype, np.float32)
        self.assertTrue(np.allclose(normalized_image, expected_image, atol=1e-06))

    def test_center_crop(self):
        if False:
            while True:
                i = 10
        image = np.random.randint(0, 256, (3, 224, 224))
        with self.assertRaises(ValueError):
            center_crop(image, 10)
        expected_image = image[:, 52:172, 82:142].transpose(1, 2, 0)
        cropped_image = center_crop(image, (120, 60), data_format='channels_last')
        self.assertIsInstance(cropped_image, np.ndarray)
        self.assertEqual(cropped_image.shape, (120, 60, 3))
        self.assertTrue(np.allclose(cropped_image, expected_image))
        expected_image = np.zeros((300, 260, 3))
        expected_image[38:262, 18:242, :] = image.transpose((1, 2, 0))
        cropped_image = center_crop(image, (300, 260), data_format='channels_last')
        self.assertIsInstance(cropped_image, np.ndarray)
        self.assertEqual(cropped_image.shape, (300, 260, 3))
        self.assertTrue(np.allclose(cropped_image, expected_image))
        image = np.random.randint(0, 256, (224, 224, 4))
        expected_image = image[52:172, 82:142, :]
        self.assertTrue(np.allclose(center_crop(image, (120, 60), input_data_format='channels_last'), expected_image))

    def test_center_to_corners_format(self):
        if False:
            print('Hello World!')
        bbox_center = np.array([[10, 20, 4, 8], [15, 16, 3, 4]])
        expected = np.array([[8, 16, 12, 24], [13.5, 14, 16.5, 18]])
        self.assertTrue(np.allclose(center_to_corners_format(bbox_center), expected))
        self.assertTrue(np.allclose(corners_to_center_format(center_to_corners_format(bbox_center)), bbox_center))

    def test_corners_to_center_format(self):
        if False:
            i = 10
            return i + 15
        bbox_corners = np.array([[8, 16, 12, 24], [13.5, 14, 16.5, 18]])
        expected = np.array([[10, 20, 4, 8], [15, 16, 3, 4]])
        self.assertTrue(np.allclose(corners_to_center_format(bbox_corners), expected))
        self.assertTrue(np.allclose(center_to_corners_format(corners_to_center_format(bbox_corners)), bbox_corners))

    def test_rgb_to_id(self):
        if False:
            while True:
                i = 10
        rgb = [125, 4, 255]
        self.assertEqual(rgb_to_id(rgb), 16712829)
        color = np.array([[[213, 54, 165], [88, 207, 39], [156, 108, 128]], [[183, 194, 46], [137, 58, 88], [114, 131, 233]]])
        expected = np.array([[10827477, 2608984, 8416412], [3064503, 5782153, 15303538]])
        self.assertTrue(np.allclose(rgb_to_id(color), expected))

    def test_id_to_rgb(self):
        if False:
            while True:
                i = 10
        self.assertEqual(id_to_rgb(16712829), [125, 4, 255])
        id_array = np.array([[10827477, 2608984, 8416412], [3064503, 5782153, 15303538]])
        color = np.array([[[213, 54, 165], [88, 207, 39], [156, 108, 128]], [[183, 194, 46], [137, 58, 88], [114, 131, 233]]])
        self.assertTrue(np.allclose(id_to_rgb(id_array), color))

    def test_pad(self):
        if False:
            i = 10
            return i + 15
        image = np.array([[[0, 1], [2, 3]]])
        with self.assertRaises(ValueError):
            pad(image, 10, mode='unknown')
        with self.assertRaises(ValueError):
            pad(image, (5, 10, 10))
        expected_image = np.array([[[0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 3, 0], [0, 0, 0, 0]]])
        self.assertTrue(np.allclose(expected_image, pad(image, 1)))
        expected_image = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 2, 3, 0], [0, 0, 0, 0, 0]])
        self.assertTrue(np.allclose(expected_image, pad(image, (2, 1))))
        expected_image = np.array([[[9, 9], [9, 9], [0, 1], [2, 3], [9, 9]]])
        self.assertTrue(np.allclose(expected_image, pad(image, ((2, 1), (0, 0)), constant_values=9)))
        expected_image = np.array([[[8, 8, 0, 1, 9], [8, 8, 2, 3, 9], [8, 8, 7, 7, 9], [8, 8, 7, 7, 9]]])
        self.assertTrue(np.allclose(expected_image, pad(image, ((0, 2), (2, 1)), constant_values=((6, 7), (8, 9)))))
        image = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]])
        expected_image = np.array([[[2, 1, 0, 1, 2, 1], [5, 4, 3, 4, 5, 4], [8, 7, 6, 7, 8, 7], [5, 4, 3, 4, 5, 4], [2, 1, 0, 1, 2, 1]]])
        self.assertTrue(np.allclose(expected_image, pad(image, ((0, 2), (2, 1)), mode='reflect')))
        expected_image = np.array([[[0, 0, 0, 1, 2, 2], [3, 3, 3, 4, 5, 5], [6, 6, 6, 7, 8, 8], [6, 6, 6, 7, 8, 8], [6, 6, 6, 7, 8, 8]]])
        self.assertTrue(np.allclose(expected_image, pad(image, ((0, 2), (2, 1)), mode='replicate')))
        expected_image = np.array([[[1, 0, 0, 1, 2, 2], [4, 3, 3, 4, 5, 5], [7, 6, 6, 7, 8, 8], [7, 6, 6, 7, 8, 8], [4, 3, 3, 4, 5, 5]]])
        self.assertTrue(np.allclose(expected_image, pad(image, ((0, 2), (2, 1)), mode='symmetric')))
        image = np.array([[[0, 1], [2, 3]]])
        expected_image = np.array([[[0], [1], [0], [1], [0]], [[2], [3], [2], [3], [2]], [[0], [1], [0], [1], [0]], [[2], [3], [2], [3], [2]]])
        self.assertTrue(np.allclose(expected_image, pad(image, ((0, 2), (2, 1)), mode='reflect', data_format='channels_last')))
        image = np.array([[[0, 1], [2, 3]]])
        expected_image = np.array([[[0, 0], [0, 1], [2, 3]], [[0, 0], [0, 0], [0, 0]]])
        self.assertTrue(np.allclose(expected_image, pad(image, ((0, 1), (1, 0)), mode='constant', input_data_format='channels_last')))

    @require_vision
    def test_convert_to_rgb(self):
        if False:
            while True:
                i = 10
        image = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.uint8)
        pil_image = PIL.Image.fromarray(image)
        self.assertEqual(pil_image.mode, 'RGBA')
        self.assertEqual(pil_image.size, (2, 1))
        rgb_image = convert_to_rgb(image)
        self.assertEqual(rgb_image.shape, (1, 2, 4))
        self.assertTrue(np.allclose(rgb_image, image))
        rgb_image = convert_to_rgb(pil_image)
        self.assertEqual(rgb_image.mode, 'RGB')
        self.assertEqual(rgb_image.size, (2, 1))
        self.assertTrue(np.allclose(np.array(rgb_image), np.array([[[1, 2, 3], [5, 6, 7]]], dtype=np.uint8)))
        image = np.array([[0, 255]], dtype=np.uint8)
        pil_image = PIL.Image.fromarray(image)
        self.assertEqual(pil_image.mode, 'L')
        self.assertEqual(pil_image.size, (2, 1))
        rgb_image = convert_to_rgb(pil_image)
        self.assertEqual(rgb_image.mode, 'RGB')
        self.assertEqual(rgb_image.size, (2, 1))
        self.assertTrue(np.allclose(np.array(rgb_image), np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)))

    def test_flip_channel_order(self):
        if False:
            return 10
        img_channels_first = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]], [[16, 17, 18, 19], [20, 21, 22, 23]]])
        img_channels_last = np.moveaxis(img_channels_first, 0, -1)
        flipped_img_channels_first = np.array([[[16, 17, 18, 19], [20, 21, 22, 23]], [[8, 9, 10, 11], [12, 13, 14, 15]], [[0, 1, 2, 3], [4, 5, 6, 7]]])
        flipped_img_channels_last = np.moveaxis(flipped_img_channels_first, 0, -1)
        self.assertTrue(np.allclose(flip_channel_order(img_channels_first), flipped_img_channels_first))
        self.assertTrue(np.allclose(flip_channel_order(img_channels_first, 'channels_last'), flipped_img_channels_last))
        self.assertTrue(np.allclose(flip_channel_order(img_channels_last), flipped_img_channels_last))
        self.assertTrue(np.allclose(flip_channel_order(img_channels_last, 'channels_first'), flipped_img_channels_first))
        img_channels_first = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]])
        flipped_img_channels_first = img_channels_first[::-1, :, :]
        self.assertTrue(np.allclose(flip_channel_order(img_channels_first, input_data_format='channels_first'), flipped_img_channels_first))