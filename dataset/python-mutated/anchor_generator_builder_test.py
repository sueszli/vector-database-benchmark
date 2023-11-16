"""Tests for anchor_generator_builder."""
import math
import tensorflow as tf
from google.protobuf import text_format
from object_detection.anchor_generators import flexible_grid_anchor_generator
from object_detection.anchor_generators import grid_anchor_generator
from object_detection.anchor_generators import multiple_grid_anchor_generator
from object_detection.anchor_generators import multiscale_grid_anchor_generator
from object_detection.builders import anchor_generator_builder
from object_detection.protos import anchor_generator_pb2

class AnchorGeneratorBuilderTest(tf.test.TestCase):

    def assert_almost_list_equal(self, expected_list, actual_list, delta=None):
        if False:
            i = 10
            return i + 15
        self.assertEqual(len(expected_list), len(actual_list))
        for (expected_item, actual_item) in zip(expected_list, actual_list):
            self.assertAlmostEqual(expected_item, actual_item, delta=delta)

    def test_build_grid_anchor_generator_with_defaults(self):
        if False:
            print('Hello World!')
        anchor_generator_text_proto = '\n      grid_anchor_generator {\n      }\n     '
        anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
        text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
        anchor_generator_object = anchor_generator_builder.build(anchor_generator_proto)
        self.assertIsInstance(anchor_generator_object, grid_anchor_generator.GridAnchorGenerator)
        self.assertListEqual(anchor_generator_object._scales, [])
        self.assertListEqual(anchor_generator_object._aspect_ratios, [])
        self.assertAllEqual(anchor_generator_object._anchor_offset, [0, 0])
        self.assertAllEqual(anchor_generator_object._anchor_stride, [16, 16])
        self.assertAllEqual(anchor_generator_object._base_anchor_size, [256, 256])

    def test_build_grid_anchor_generator_with_non_default_parameters(self):
        if False:
            while True:
                i = 10
        anchor_generator_text_proto = '\n      grid_anchor_generator {\n        height: 128\n        width: 512\n        height_stride: 10\n        width_stride: 20\n        height_offset: 30\n        width_offset: 40\n        scales: [0.4, 2.2]\n        aspect_ratios: [0.3, 4.5]\n      }\n     '
        anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
        text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
        anchor_generator_object = anchor_generator_builder.build(anchor_generator_proto)
        self.assertIsInstance(anchor_generator_object, grid_anchor_generator.GridAnchorGenerator)
        self.assert_almost_list_equal(anchor_generator_object._scales, [0.4, 2.2])
        self.assert_almost_list_equal(anchor_generator_object._aspect_ratios, [0.3, 4.5])
        self.assertAllEqual(anchor_generator_object._anchor_offset, [30, 40])
        self.assertAllEqual(anchor_generator_object._anchor_stride, [10, 20])
        self.assertAllEqual(anchor_generator_object._base_anchor_size, [128, 512])

    def test_build_ssd_anchor_generator_with_defaults(self):
        if False:
            for i in range(10):
                print('nop')
        anchor_generator_text_proto = '\n      ssd_anchor_generator {\n        aspect_ratios: [1.0]\n      }\n    '
        anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
        text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
        anchor_generator_object = anchor_generator_builder.build(anchor_generator_proto)
        self.assertIsInstance(anchor_generator_object, multiple_grid_anchor_generator.MultipleGridAnchorGenerator)
        for (actual_scales, expected_scales) in zip(list(anchor_generator_object._scales), [(0.1, 0.2, 0.2), (0.35, 0.418), (0.499, 0.57), (0.649, 0.721), (0.799, 0.871), (0.949, 0.974)]):
            self.assert_almost_list_equal(expected_scales, actual_scales, delta=0.01)
        for (actual_aspect_ratio, expected_aspect_ratio) in zip(list(anchor_generator_object._aspect_ratios), [(1.0, 2.0, 0.5)] + 5 * [(1.0, 1.0)]):
            self.assert_almost_list_equal(expected_aspect_ratio, actual_aspect_ratio)
        self.assertAllClose(anchor_generator_object._base_anchor_size, [1.0, 1.0])

    def test_build_ssd_anchor_generator_with_custom_scales(self):
        if False:
            return 10
        anchor_generator_text_proto = '\n      ssd_anchor_generator {\n        aspect_ratios: [1.0]\n        scales: [0.1, 0.15, 0.2, 0.4, 0.6, 0.8]\n        reduce_boxes_in_lowest_layer: false\n      }\n    '
        anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
        text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
        anchor_generator_object = anchor_generator_builder.build(anchor_generator_proto)
        self.assertIsInstance(anchor_generator_object, multiple_grid_anchor_generator.MultipleGridAnchorGenerator)
        for (actual_scales, expected_scales) in zip(list(anchor_generator_object._scales), [(0.1, math.sqrt(0.1 * 0.15)), (0.15, math.sqrt(0.15 * 0.2)), (0.2, math.sqrt(0.2 * 0.4)), (0.4, math.sqrt(0.4 * 0.6)), (0.6, math.sqrt(0.6 * 0.8)), (0.8, math.sqrt(0.8 * 1.0))]):
            self.assert_almost_list_equal(expected_scales, actual_scales, delta=0.01)

    def test_build_ssd_anchor_generator_with_custom_interpolated_scale(self):
        if False:
            for i in range(10):
                print('nop')
        anchor_generator_text_proto = '\n      ssd_anchor_generator {\n        aspect_ratios: [0.5]\n        interpolated_scale_aspect_ratio: 0.5\n        reduce_boxes_in_lowest_layer: false\n      }\n    '
        anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
        text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
        anchor_generator_object = anchor_generator_builder.build(anchor_generator_proto)
        self.assertIsInstance(anchor_generator_object, multiple_grid_anchor_generator.MultipleGridAnchorGenerator)
        for (actual_aspect_ratio, expected_aspect_ratio) in zip(list(anchor_generator_object._aspect_ratios), 6 * [(0.5, 0.5)]):
            self.assert_almost_list_equal(expected_aspect_ratio, actual_aspect_ratio)

    def test_build_ssd_anchor_generator_without_reduced_boxes(self):
        if False:
            for i in range(10):
                print('nop')
        anchor_generator_text_proto = '\n      ssd_anchor_generator {\n        aspect_ratios: [1.0]\n        reduce_boxes_in_lowest_layer: false\n      }\n    '
        anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
        text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
        anchor_generator_object = anchor_generator_builder.build(anchor_generator_proto)
        self.assertIsInstance(anchor_generator_object, multiple_grid_anchor_generator.MultipleGridAnchorGenerator)
        for (actual_scales, expected_scales) in zip(list(anchor_generator_object._scales), [(0.2, 0.264), (0.35, 0.418), (0.499, 0.57), (0.649, 0.721), (0.799, 0.871), (0.949, 0.974)]):
            self.assert_almost_list_equal(expected_scales, actual_scales, delta=0.01)
        for (actual_aspect_ratio, expected_aspect_ratio) in zip(list(anchor_generator_object._aspect_ratios), 6 * [(1.0, 1.0)]):
            self.assert_almost_list_equal(expected_aspect_ratio, actual_aspect_ratio)
        self.assertAllClose(anchor_generator_object._base_anchor_size, [1.0, 1.0])

    def test_build_ssd_anchor_generator_with_non_default_parameters(self):
        if False:
            while True:
                i = 10
        anchor_generator_text_proto = '\n      ssd_anchor_generator {\n        num_layers: 2\n        min_scale: 0.3\n        max_scale: 0.8\n        aspect_ratios: [2.0]\n        height_stride: 16\n        height_stride: 32\n        width_stride: 20\n        width_stride: 30\n        height_offset: 8\n        height_offset: 16\n        width_offset: 0\n        width_offset: 10\n      }\n    '
        anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
        text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
        anchor_generator_object = anchor_generator_builder.build(anchor_generator_proto)
        self.assertIsInstance(anchor_generator_object, multiple_grid_anchor_generator.MultipleGridAnchorGenerator)
        for (actual_scales, expected_scales) in zip(list(anchor_generator_object._scales), [(0.1, 0.3, 0.3), (0.8, 0.894)]):
            self.assert_almost_list_equal(expected_scales, actual_scales, delta=0.01)
        for (actual_aspect_ratio, expected_aspect_ratio) in zip(list(anchor_generator_object._aspect_ratios), [(1.0, 2.0, 0.5), (2.0, 1.0)]):
            self.assert_almost_list_equal(expected_aspect_ratio, actual_aspect_ratio)
        for (actual_strides, expected_strides) in zip(list(anchor_generator_object._anchor_strides), [(16, 20), (32, 30)]):
            self.assert_almost_list_equal(expected_strides, actual_strides)
        for (actual_offsets, expected_offsets) in zip(list(anchor_generator_object._anchor_offsets), [(8, 0), (16, 10)]):
            self.assert_almost_list_equal(expected_offsets, actual_offsets)
        self.assertAllClose(anchor_generator_object._base_anchor_size, [1.0, 1.0])

    def test_raise_value_error_on_empty_anchor_genertor(self):
        if False:
            print('Hello World!')
        anchor_generator_text_proto = '\n    '
        anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
        text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
        with self.assertRaises(ValueError):
            anchor_generator_builder.build(anchor_generator_proto)

    def test_build_multiscale_anchor_generator_custom_aspect_ratios(self):
        if False:
            print('Hello World!')
        anchor_generator_text_proto = '\n      multiscale_anchor_generator {\n        aspect_ratios: [1.0]\n      }\n    '
        anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
        text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
        anchor_generator_object = anchor_generator_builder.build(anchor_generator_proto)
        self.assertIsInstance(anchor_generator_object, multiscale_grid_anchor_generator.MultiscaleGridAnchorGenerator)
        for (level, anchor_grid_info) in zip(range(3, 8), anchor_generator_object._anchor_grid_info):
            self.assertEqual(set(anchor_grid_info.keys()), set(['level', 'info']))
            self.assertTrue(level, anchor_grid_info['level'])
            self.assertEqual(len(anchor_grid_info['info']), 4)
            self.assertAllClose(anchor_grid_info['info'][0], [2 ** 0, 2 ** 0.5])
            self.assertTrue(anchor_grid_info['info'][1], 1.0)
            self.assertAllClose(anchor_grid_info['info'][2], [4.0 * 2 ** level, 4.0 * 2 ** level])
            self.assertAllClose(anchor_grid_info['info'][3], [2 ** level, 2 ** level])
            self.assertTrue(anchor_generator_object._normalize_coordinates)

    def test_build_multiscale_anchor_generator_with_anchors_in_pixel_coordinates(self):
        if False:
            print('Hello World!')
        anchor_generator_text_proto = '\n      multiscale_anchor_generator {\n        aspect_ratios: [1.0]\n        normalize_coordinates: false\n      }\n    '
        anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
        text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
        anchor_generator_object = anchor_generator_builder.build(anchor_generator_proto)
        self.assertIsInstance(anchor_generator_object, multiscale_grid_anchor_generator.MultiscaleGridAnchorGenerator)
        self.assertFalse(anchor_generator_object._normalize_coordinates)

    def test_build_flexible_anchor_generator(self):
        if False:
            while True:
                i = 10
        anchor_generator_text_proto = '\n      flexible_grid_anchor_generator {\n        anchor_grid {\n          base_sizes: [1.5]\n          aspect_ratios: [1.0]\n          height_stride: 16\n          width_stride: 20\n          height_offset: 8\n          width_offset: 9\n        }\n        anchor_grid {\n          base_sizes: [1.0, 2.0]\n          aspect_ratios: [1.0, 0.5]\n          height_stride: 32\n          width_stride: 30\n          height_offset: 10\n          width_offset: 11\n        }\n      }\n    '
        anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
        text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
        anchor_generator_object = anchor_generator_builder.build(anchor_generator_proto)
        self.assertIsInstance(anchor_generator_object, flexible_grid_anchor_generator.FlexibleGridAnchorGenerator)
        for (actual_base_sizes, expected_base_sizes) in zip(list(anchor_generator_object._base_sizes), [(1.5,), (1.0, 2.0)]):
            self.assert_almost_list_equal(expected_base_sizes, actual_base_sizes)
        for (actual_aspect_ratios, expected_aspect_ratios) in zip(list(anchor_generator_object._aspect_ratios), [(1.0,), (1.0, 0.5)]):
            self.assert_almost_list_equal(expected_aspect_ratios, actual_aspect_ratios)
        for (actual_strides, expected_strides) in zip(list(anchor_generator_object._anchor_strides), [(16, 20), (32, 30)]):
            self.assert_almost_list_equal(expected_strides, actual_strides)
        for (actual_offsets, expected_offsets) in zip(list(anchor_generator_object._anchor_offsets), [(8, 9), (10, 11)]):
            self.assert_almost_list_equal(expected_offsets, actual_offsets)
        self.assertTrue(anchor_generator_object._normalize_coordinates)
if __name__ == '__main__':
    tf.test.main()