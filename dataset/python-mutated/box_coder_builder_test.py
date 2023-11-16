"""Tests for box_coder_builder."""
import tensorflow as tf
from google.protobuf import text_format
from object_detection.box_coders import faster_rcnn_box_coder
from object_detection.box_coders import keypoint_box_coder
from object_detection.box_coders import mean_stddev_box_coder
from object_detection.box_coders import square_box_coder
from object_detection.builders import box_coder_builder
from object_detection.protos import box_coder_pb2

class BoxCoderBuilderTest(tf.test.TestCase):

    def test_build_faster_rcnn_box_coder_with_defaults(self):
        if False:
            return 10
        box_coder_text_proto = '\n      faster_rcnn_box_coder {\n      }\n    '
        box_coder_proto = box_coder_pb2.BoxCoder()
        text_format.Merge(box_coder_text_proto, box_coder_proto)
        box_coder_object = box_coder_builder.build(box_coder_proto)
        self.assertIsInstance(box_coder_object, faster_rcnn_box_coder.FasterRcnnBoxCoder)
        self.assertEqual(box_coder_object._scale_factors, [10.0, 10.0, 5.0, 5.0])

    def test_build_faster_rcnn_box_coder_with_non_default_parameters(self):
        if False:
            i = 10
            return i + 15
        box_coder_text_proto = '\n      faster_rcnn_box_coder {\n        y_scale: 6.0\n        x_scale: 3.0\n        height_scale: 7.0\n        width_scale: 8.0\n      }\n    '
        box_coder_proto = box_coder_pb2.BoxCoder()
        text_format.Merge(box_coder_text_proto, box_coder_proto)
        box_coder_object = box_coder_builder.build(box_coder_proto)
        self.assertIsInstance(box_coder_object, faster_rcnn_box_coder.FasterRcnnBoxCoder)
        self.assertEqual(box_coder_object._scale_factors, [6.0, 3.0, 7.0, 8.0])

    def test_build_keypoint_box_coder_with_defaults(self):
        if False:
            print('Hello World!')
        box_coder_text_proto = '\n      keypoint_box_coder {\n      }\n    '
        box_coder_proto = box_coder_pb2.BoxCoder()
        text_format.Merge(box_coder_text_proto, box_coder_proto)
        box_coder_object = box_coder_builder.build(box_coder_proto)
        self.assertIsInstance(box_coder_object, keypoint_box_coder.KeypointBoxCoder)
        self.assertEqual(box_coder_object._scale_factors, [10.0, 10.0, 5.0, 5.0])

    def test_build_keypoint_box_coder_with_non_default_parameters(self):
        if False:
            print('Hello World!')
        box_coder_text_proto = '\n      keypoint_box_coder {\n        num_keypoints: 6\n        y_scale: 6.0\n        x_scale: 3.0\n        height_scale: 7.0\n        width_scale: 8.0\n      }\n    '
        box_coder_proto = box_coder_pb2.BoxCoder()
        text_format.Merge(box_coder_text_proto, box_coder_proto)
        box_coder_object = box_coder_builder.build(box_coder_proto)
        self.assertIsInstance(box_coder_object, keypoint_box_coder.KeypointBoxCoder)
        self.assertEqual(box_coder_object._num_keypoints, 6)
        self.assertEqual(box_coder_object._scale_factors, [6.0, 3.0, 7.0, 8.0])

    def test_build_mean_stddev_box_coder(self):
        if False:
            i = 10
            return i + 15
        box_coder_text_proto = '\n      mean_stddev_box_coder {\n      }\n    '
        box_coder_proto = box_coder_pb2.BoxCoder()
        text_format.Merge(box_coder_text_proto, box_coder_proto)
        box_coder_object = box_coder_builder.build(box_coder_proto)
        self.assertTrue(isinstance(box_coder_object, mean_stddev_box_coder.MeanStddevBoxCoder))

    def test_build_square_box_coder_with_defaults(self):
        if False:
            i = 10
            return i + 15
        box_coder_text_proto = '\n      square_box_coder {\n      }\n    '
        box_coder_proto = box_coder_pb2.BoxCoder()
        text_format.Merge(box_coder_text_proto, box_coder_proto)
        box_coder_object = box_coder_builder.build(box_coder_proto)
        self.assertTrue(isinstance(box_coder_object, square_box_coder.SquareBoxCoder))
        self.assertEqual(box_coder_object._scale_factors, [10.0, 10.0, 5.0])

    def test_build_square_box_coder_with_non_default_parameters(self):
        if False:
            i = 10
            return i + 15
        box_coder_text_proto = '\n      square_box_coder {\n        y_scale: 6.0\n        x_scale: 3.0\n        length_scale: 7.0\n      }\n    '
        box_coder_proto = box_coder_pb2.BoxCoder()
        text_format.Merge(box_coder_text_proto, box_coder_proto)
        box_coder_object = box_coder_builder.build(box_coder_proto)
        self.assertTrue(isinstance(box_coder_object, square_box_coder.SquareBoxCoder))
        self.assertEqual(box_coder_object._scale_factors, [6.0, 3.0, 7.0])

    def test_raise_error_on_empty_box_coder(self):
        if False:
            while True:
                i = 10
        box_coder_text_proto = '\n    '
        box_coder_proto = box_coder_pb2.BoxCoder()
        text_format.Merge(box_coder_text_proto, box_coder_proto)
        with self.assertRaises(ValueError):
            box_coder_builder.build(box_coder_proto)
if __name__ == '__main__':
    tf.test.main()