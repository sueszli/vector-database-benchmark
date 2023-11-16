"""Tests for preprocessor_builder."""
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import preprocessor_builder
from object_detection.core import preprocessor
from object_detection.protos import preprocessor_pb2

class PreprocessorBuilderTest(tf.test.TestCase):

    def assert_dictionary_close(self, dict1, dict2):
        if False:
            print('Hello World!')
        'Helper to check if two dicts with floatst or integers are close.'
        self.assertEqual(sorted(dict1.keys()), sorted(dict2.keys()))
        for key in dict1:
            value = dict1[key]
            if isinstance(value, float):
                self.assertAlmostEqual(value, dict2[key])
            else:
                self.assertEqual(value, dict2[key])

    def test_build_normalize_image(self):
        if False:
            return 10
        preprocessor_text_proto = '\n    normalize_image {\n      original_minval: 0.0\n      original_maxval: 255.0\n      target_minval: -1.0\n      target_maxval: 1.0\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.normalize_image)
        self.assertEqual(args, {'original_minval': 0.0, 'original_maxval': 255.0, 'target_minval': -1.0, 'target_maxval': 1.0})

    def test_build_random_horizontal_flip(self):
        if False:
            return 10
        preprocessor_text_proto = '\n    random_horizontal_flip {\n      keypoint_flip_permutation: 1\n      keypoint_flip_permutation: 0\n      keypoint_flip_permutation: 2\n      keypoint_flip_permutation: 3\n      keypoint_flip_permutation: 5\n      keypoint_flip_permutation: 4\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_horizontal_flip)
        self.assertEqual(args, {'keypoint_flip_permutation': (1, 0, 2, 3, 5, 4)})

    def test_build_random_vertical_flip(self):
        if False:
            for i in range(10):
                print('nop')
        preprocessor_text_proto = '\n    random_vertical_flip {\n      keypoint_flip_permutation: 1\n      keypoint_flip_permutation: 0\n      keypoint_flip_permutation: 2\n      keypoint_flip_permutation: 3\n      keypoint_flip_permutation: 5\n      keypoint_flip_permutation: 4\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_vertical_flip)
        self.assertEqual(args, {'keypoint_flip_permutation': (1, 0, 2, 3, 5, 4)})

    def test_build_random_rotation90(self):
        if False:
            while True:
                i = 10
        preprocessor_text_proto = '\n    random_rotation90 {}\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_rotation90)
        self.assertEqual(args, {})

    def test_build_random_pixel_value_scale(self):
        if False:
            return 10
        preprocessor_text_proto = '\n    random_pixel_value_scale {\n      minval: 0.8\n      maxval: 1.2\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_pixel_value_scale)
        self.assert_dictionary_close(args, {'minval': 0.8, 'maxval': 1.2})

    def test_build_random_image_scale(self):
        if False:
            while True:
                i = 10
        preprocessor_text_proto = '\n    random_image_scale {\n      min_scale_ratio: 0.8\n      max_scale_ratio: 2.2\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_image_scale)
        self.assert_dictionary_close(args, {'min_scale_ratio': 0.8, 'max_scale_ratio': 2.2})

    def test_build_random_rgb_to_gray(self):
        if False:
            return 10
        preprocessor_text_proto = '\n    random_rgb_to_gray {\n      probability: 0.8\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_rgb_to_gray)
        self.assert_dictionary_close(args, {'probability': 0.8})

    def test_build_random_adjust_brightness(self):
        if False:
            for i in range(10):
                print('nop')
        preprocessor_text_proto = '\n    random_adjust_brightness {\n      max_delta: 0.2\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_adjust_brightness)
        self.assert_dictionary_close(args, {'max_delta': 0.2})

    def test_build_random_adjust_contrast(self):
        if False:
            print('Hello World!')
        preprocessor_text_proto = '\n    random_adjust_contrast {\n      min_delta: 0.7\n      max_delta: 1.1\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_adjust_contrast)
        self.assert_dictionary_close(args, {'min_delta': 0.7, 'max_delta': 1.1})

    def test_build_random_adjust_hue(self):
        if False:
            i = 10
            return i + 15
        preprocessor_text_proto = '\n    random_adjust_hue {\n      max_delta: 0.01\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_adjust_hue)
        self.assert_dictionary_close(args, {'max_delta': 0.01})

    def test_build_random_adjust_saturation(self):
        if False:
            return 10
        preprocessor_text_proto = '\n    random_adjust_saturation {\n      min_delta: 0.75\n      max_delta: 1.15\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_adjust_saturation)
        self.assert_dictionary_close(args, {'min_delta': 0.75, 'max_delta': 1.15})

    def test_build_random_distort_color(self):
        if False:
            i = 10
            return i + 15
        preprocessor_text_proto = '\n    random_distort_color {\n      color_ordering: 1\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_distort_color)
        self.assertEqual(args, {'color_ordering': 1})

    def test_build_random_jitter_boxes(self):
        if False:
            print('Hello World!')
        preprocessor_text_proto = '\n    random_jitter_boxes {\n      ratio: 0.1\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_jitter_boxes)
        self.assert_dictionary_close(args, {'ratio': 0.1})

    def test_build_random_crop_image(self):
        if False:
            return 10
        preprocessor_text_proto = '\n    random_crop_image {\n      min_object_covered: 0.75\n      min_aspect_ratio: 0.75\n      max_aspect_ratio: 1.5\n      min_area: 0.25\n      max_area: 0.875\n      overlap_thresh: 0.5\n      clip_boxes: False\n      random_coef: 0.125\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_crop_image)
        self.assertEqual(args, {'min_object_covered': 0.75, 'aspect_ratio_range': (0.75, 1.5), 'area_range': (0.25, 0.875), 'overlap_thresh': 0.5, 'clip_boxes': False, 'random_coef': 0.125})

    def test_build_random_pad_image(self):
        if False:
            print('Hello World!')
        preprocessor_text_proto = '\n    random_pad_image {\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_pad_image)
        self.assertEqual(args, {'min_image_size': None, 'max_image_size': None, 'pad_color': None})

    def test_build_random_absolute_pad_image(self):
        if False:
            print('Hello World!')
        preprocessor_text_proto = '\n    random_absolute_pad_image {\n      max_height_padding: 50\n      max_width_padding: 100\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_absolute_pad_image)
        self.assertEqual(args, {'max_height_padding': 50, 'max_width_padding': 100, 'pad_color': None})

    def test_build_random_crop_pad_image(self):
        if False:
            i = 10
            return i + 15
        preprocessor_text_proto = '\n    random_crop_pad_image {\n      min_object_covered: 0.75\n      min_aspect_ratio: 0.75\n      max_aspect_ratio: 1.5\n      min_area: 0.25\n      max_area: 0.875\n      overlap_thresh: 0.5\n      clip_boxes: False\n      random_coef: 0.125\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_crop_pad_image)
        self.assertEqual(args, {'min_object_covered': 0.75, 'aspect_ratio_range': (0.75, 1.5), 'area_range': (0.25, 0.875), 'overlap_thresh': 0.5, 'clip_boxes': False, 'random_coef': 0.125, 'pad_color': None})

    def test_build_random_crop_pad_image_with_optional_parameters(self):
        if False:
            i = 10
            return i + 15
        preprocessor_text_proto = '\n    random_crop_pad_image {\n      min_object_covered: 0.75\n      min_aspect_ratio: 0.75\n      max_aspect_ratio: 1.5\n      min_area: 0.25\n      max_area: 0.875\n      overlap_thresh: 0.5\n      clip_boxes: False\n      random_coef: 0.125\n      min_padded_size_ratio: 0.5\n      min_padded_size_ratio: 0.75\n      max_padded_size_ratio: 0.5\n      max_padded_size_ratio: 0.75\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_crop_pad_image)
        self.assertEqual(args, {'min_object_covered': 0.75, 'aspect_ratio_range': (0.75, 1.5), 'area_range': (0.25, 0.875), 'overlap_thresh': 0.5, 'clip_boxes': False, 'random_coef': 0.125, 'min_padded_size_ratio': (0.5, 0.75), 'max_padded_size_ratio': (0.5, 0.75), 'pad_color': None})

    def test_build_random_crop_to_aspect_ratio(self):
        if False:
            print('Hello World!')
        preprocessor_text_proto = '\n    random_crop_to_aspect_ratio {\n      aspect_ratio: 0.85\n      overlap_thresh: 0.35\n      clip_boxes: False\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_crop_to_aspect_ratio)
        self.assert_dictionary_close(args, {'aspect_ratio': 0.85, 'overlap_thresh': 0.35, 'clip_boxes': False})

    def test_build_random_black_patches(self):
        if False:
            for i in range(10):
                print('nop')
        preprocessor_text_proto = '\n    random_black_patches {\n      max_black_patches: 20\n      probability: 0.95\n      size_to_image_ratio: 0.12\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_black_patches)
        self.assert_dictionary_close(args, {'max_black_patches': 20, 'probability': 0.95, 'size_to_image_ratio': 0.12})

    def test_build_random_jpeg_quality(self):
        if False:
            print('Hello World!')
        preprocessor_text_proto = '\n    random_jpeg_quality {\n      random_coef: 0.5\n      min_jpeg_quality: 40\n      max_jpeg_quality: 90\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Parse(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_jpeg_quality)
        self.assert_dictionary_close(args, {'random_coef': 0.5, 'min_jpeg_quality': 40, 'max_jpeg_quality': 90})

    def test_build_random_downscale_to_target_pixels(self):
        if False:
            i = 10
            return i + 15
        preprocessor_text_proto = '\n    random_downscale_to_target_pixels {\n      random_coef: 0.5\n      min_target_pixels: 200\n      max_target_pixels: 900\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Parse(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_downscale_to_target_pixels)
        self.assert_dictionary_close(args, {'random_coef': 0.5, 'min_target_pixels': 200, 'max_target_pixels': 900})

    def test_build_random_patch_gaussian(self):
        if False:
            i = 10
            return i + 15
        preprocessor_text_proto = '\n    random_patch_gaussian {\n      random_coef: 0.5\n      min_patch_size: 10\n      max_patch_size: 300\n      min_gaussian_stddev: 0.2\n      max_gaussian_stddev: 1.5\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Parse(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_patch_gaussian)
        self.assert_dictionary_close(args, {'random_coef': 0.5, 'min_patch_size': 10, 'max_patch_size': 300, 'min_gaussian_stddev': 0.2, 'max_gaussian_stddev': 1.5})

    def test_auto_augment_image(self):
        if False:
            i = 10
            return i + 15
        preprocessor_text_proto = "\n    autoaugment_image {\n      policy_name: 'v0'\n    }\n    "
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.autoaugment_image)
        self.assert_dictionary_close(args, {'policy_name': 'v0'})

    def test_drop_label_probabilistically(self):
        if False:
            return 10
        preprocessor_text_proto = '\n    drop_label_probabilistically{\n      label: 2\n      drop_probability: 0.5\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.drop_label_probabilistically)
        self.assert_dictionary_close(args, {'dropped_label': 2, 'drop_probability': 0.5})

    def test_remap_labels(self):
        if False:
            for i in range(10):
                print('nop')
        preprocessor_text_proto = '\n    remap_labels{\n      original_labels: 1\n      original_labels: 2\n      new_label: 3\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.remap_labels)
        self.assert_dictionary_close(args, {'original_labels': [1, 2], 'new_label': 3})

    def test_build_random_resize_method(self):
        if False:
            return 10
        preprocessor_text_proto = '\n    random_resize_method {\n      target_height: 75\n      target_width: 100\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_resize_method)
        self.assert_dictionary_close(args, {'target_size': [75, 100]})

    def test_build_scale_boxes_to_pixel_coordinates(self):
        if False:
            while True:
                i = 10
        preprocessor_text_proto = '\n    scale_boxes_to_pixel_coordinates {}\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.scale_boxes_to_pixel_coordinates)
        self.assertEqual(args, {})

    def test_build_resize_image(self):
        if False:
            while True:
                i = 10
        preprocessor_text_proto = '\n    resize_image {\n      new_height: 75\n      new_width: 100\n      method: BICUBIC\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.resize_image)
        self.assertEqual(args, {'new_height': 75, 'new_width': 100, 'method': tf.image.ResizeMethod.BICUBIC})

    def test_build_rgb_to_gray(self):
        if False:
            i = 10
            return i + 15
        preprocessor_text_proto = '\n    rgb_to_gray {}\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.rgb_to_gray)
        self.assertEqual(args, {})

    def test_build_subtract_channel_mean(self):
        if False:
            while True:
                i = 10
        preprocessor_text_proto = '\n    subtract_channel_mean {\n      means: [1.0, 2.0, 3.0]\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.subtract_channel_mean)
        self.assertEqual(args, {'means': [1.0, 2.0, 3.0]})

    def test_random_self_concat_image(self):
        if False:
            for i in range(10):
                print('nop')
        preprocessor_text_proto = '\n    random_self_concat_image {\n      concat_vertical_probability: 0.5\n      concat_horizontal_probability: 0.25\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.random_self_concat_image)
        self.assertEqual(args, {'concat_vertical_probability': 0.5, 'concat_horizontal_probability': 0.25})

    def test_build_ssd_random_crop(self):
        if False:
            i = 10
            return i + 15
        preprocessor_text_proto = '\n    ssd_random_crop {\n      operations {\n        min_object_covered: 0.0\n        min_aspect_ratio: 0.875\n        max_aspect_ratio: 1.125\n        min_area: 0.5\n        max_area: 1.0\n        overlap_thresh: 0.0\n        clip_boxes: False\n        random_coef: 0.375\n      }\n      operations {\n        min_object_covered: 0.25\n        min_aspect_ratio: 0.75\n        max_aspect_ratio: 1.5\n        min_area: 0.5\n        max_area: 1.0\n        overlap_thresh: 0.25\n        clip_boxes: True\n        random_coef: 0.375\n      }\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.ssd_random_crop)
        self.assertEqual(args, {'min_object_covered': [0.0, 0.25], 'aspect_ratio_range': [(0.875, 1.125), (0.75, 1.5)], 'area_range': [(0.5, 1.0), (0.5, 1.0)], 'overlap_thresh': [0.0, 0.25], 'clip_boxes': [False, True], 'random_coef': [0.375, 0.375]})

    def test_build_ssd_random_crop_empty_operations(self):
        if False:
            while True:
                i = 10
        preprocessor_text_proto = '\n    ssd_random_crop {\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.ssd_random_crop)
        self.assertEqual(args, {})

    def test_build_ssd_random_crop_pad(self):
        if False:
            for i in range(10):
                print('nop')
        preprocessor_text_proto = '\n    ssd_random_crop_pad {\n      operations {\n        min_object_covered: 0.0\n        min_aspect_ratio: 0.875\n        max_aspect_ratio: 1.125\n        min_area: 0.5\n        max_area: 1.0\n        overlap_thresh: 0.0\n        clip_boxes: False\n        random_coef: 0.375\n        min_padded_size_ratio: [1.0, 1.0]\n        max_padded_size_ratio: [2.0, 2.0]\n        pad_color_r: 0.5\n        pad_color_g: 0.5\n        pad_color_b: 0.5\n      }\n      operations {\n        min_object_covered: 0.25\n        min_aspect_ratio: 0.75\n        max_aspect_ratio: 1.5\n        min_area: 0.5\n        max_area: 1.0\n        overlap_thresh: 0.25\n        clip_boxes: True\n        random_coef: 0.375\n        min_padded_size_ratio: [1.0, 1.0]\n        max_padded_size_ratio: [2.0, 2.0]\n        pad_color_r: 0.5\n        pad_color_g: 0.5\n        pad_color_b: 0.5\n      }\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.ssd_random_crop_pad)
        self.assertEqual(args, {'min_object_covered': [0.0, 0.25], 'aspect_ratio_range': [(0.875, 1.125), (0.75, 1.5)], 'area_range': [(0.5, 1.0), (0.5, 1.0)], 'overlap_thresh': [0.0, 0.25], 'clip_boxes': [False, True], 'random_coef': [0.375, 0.375], 'min_padded_size_ratio': [(1.0, 1.0), (1.0, 1.0)], 'max_padded_size_ratio': [(2.0, 2.0), (2.0, 2.0)], 'pad_color': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]})

    def test_build_ssd_random_crop_fixed_aspect_ratio(self):
        if False:
            i = 10
            return i + 15
        preprocessor_text_proto = '\n    ssd_random_crop_fixed_aspect_ratio {\n      operations {\n        min_object_covered: 0.0\n        min_area: 0.5\n        max_area: 1.0\n        overlap_thresh: 0.0\n        clip_boxes: False\n        random_coef: 0.375\n      }\n      operations {\n        min_object_covered: 0.25\n        min_area: 0.5\n        max_area: 1.0\n        overlap_thresh: 0.25\n        clip_boxes: True\n        random_coef: 0.375\n      }\n      aspect_ratio: 0.875\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.ssd_random_crop_fixed_aspect_ratio)
        self.assertEqual(args, {'min_object_covered': [0.0, 0.25], 'aspect_ratio': 0.875, 'area_range': [(0.5, 1.0), (0.5, 1.0)], 'overlap_thresh': [0.0, 0.25], 'clip_boxes': [False, True], 'random_coef': [0.375, 0.375]})

    def test_build_ssd_random_crop_pad_fixed_aspect_ratio(self):
        if False:
            print('Hello World!')
        preprocessor_text_proto = '\n    ssd_random_crop_pad_fixed_aspect_ratio {\n      operations {\n        min_object_covered: 0.0\n        min_aspect_ratio: 0.875\n        max_aspect_ratio: 1.125\n        min_area: 0.5\n        max_area: 1.0\n        overlap_thresh: 0.0\n        clip_boxes: False\n        random_coef: 0.375\n      }\n      operations {\n        min_object_covered: 0.25\n        min_aspect_ratio: 0.75\n        max_aspect_ratio: 1.5\n        min_area: 0.5\n        max_area: 1.0\n        overlap_thresh: 0.25\n        clip_boxes: True\n        random_coef: 0.375\n      }\n      aspect_ratio: 0.875\n      min_padded_size_ratio: [1.0, 1.0]\n      max_padded_size_ratio: [2.0, 2.0]\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.ssd_random_crop_pad_fixed_aspect_ratio)
        self.assertEqual(args, {'min_object_covered': [0.0, 0.25], 'aspect_ratio': 0.875, 'aspect_ratio_range': [(0.875, 1.125), (0.75, 1.5)], 'area_range': [(0.5, 1.0), (0.5, 1.0)], 'overlap_thresh': [0.0, 0.25], 'clip_boxes': [False, True], 'random_coef': [0.375, 0.375], 'min_padded_size_ratio': (1.0, 1.0), 'max_padded_size_ratio': (2.0, 2.0)})

    def test_build_normalize_image_convert_class_logits_to_softmax(self):
        if False:
            i = 10
            return i + 15
        preprocessor_text_proto = '\n    convert_class_logits_to_softmax {\n        temperature: 2\n    }\n    '
        preprocessor_proto = preprocessor_pb2.PreprocessingStep()
        text_format.Merge(preprocessor_text_proto, preprocessor_proto)
        (function, args) = preprocessor_builder.build(preprocessor_proto)
        self.assertEqual(function, preprocessor.convert_class_logits_to_softmax)
        self.assertEqual(args, {'temperature': 2})
if __name__ == '__main__':
    tf.test.main()