"""Tests for post_processing_builder."""
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import post_processing_builder
from object_detection.protos import post_processing_pb2

class PostProcessingBuilderTest(tf.test.TestCase):

    def test_build_non_max_suppressor_with_correct_parameters(self):
        if False:
            while True:
                i = 10
        post_processing_text_proto = '\n      batch_non_max_suppression {\n        score_threshold: 0.7\n        iou_threshold: 0.6\n        max_detections_per_class: 100\n        max_total_detections: 300\n        soft_nms_sigma: 0.4\n      }\n    '
        post_processing_config = post_processing_pb2.PostProcessing()
        text_format.Merge(post_processing_text_proto, post_processing_config)
        (non_max_suppressor, _) = post_processing_builder.build(post_processing_config)
        self.assertEqual(non_max_suppressor.keywords['max_size_per_class'], 100)
        self.assertEqual(non_max_suppressor.keywords['max_total_size'], 300)
        self.assertAlmostEqual(non_max_suppressor.keywords['score_thresh'], 0.7)
        self.assertAlmostEqual(non_max_suppressor.keywords['iou_thresh'], 0.6)
        self.assertAlmostEqual(non_max_suppressor.keywords['soft_nms_sigma'], 0.4)

    def test_build_non_max_suppressor_with_correct_parameters_classagnostic_nms(self):
        if False:
            while True:
                i = 10
        post_processing_text_proto = '\n      batch_non_max_suppression {\n        score_threshold: 0.7\n        iou_threshold: 0.6\n        max_detections_per_class: 10\n        max_total_detections: 300\n        use_class_agnostic_nms: True\n        max_classes_per_detection: 1\n      }\n    '
        post_processing_config = post_processing_pb2.PostProcessing()
        text_format.Merge(post_processing_text_proto, post_processing_config)
        (non_max_suppressor, _) = post_processing_builder.build(post_processing_config)
        self.assertEqual(non_max_suppressor.keywords['max_size_per_class'], 10)
        self.assertEqual(non_max_suppressor.keywords['max_total_size'], 300)
        self.assertEqual(non_max_suppressor.keywords['max_classes_per_detection'], 1)
        self.assertEqual(non_max_suppressor.keywords['use_class_agnostic_nms'], True)
        self.assertAlmostEqual(non_max_suppressor.keywords['score_thresh'], 0.7)
        self.assertAlmostEqual(non_max_suppressor.keywords['iou_thresh'], 0.6)

    def test_build_identity_score_converter(self):
        if False:
            for i in range(10):
                print('nop')
        post_processing_text_proto = '\n      score_converter: IDENTITY\n    '
        post_processing_config = post_processing_pb2.PostProcessing()
        text_format.Merge(post_processing_text_proto, post_processing_config)
        (_, score_converter) = post_processing_builder.build(post_processing_config)
        self.assertEqual(score_converter.__name__, 'identity_with_logit_scale')
        inputs = tf.constant([1, 1], tf.float32)
        outputs = score_converter(inputs)
        with self.test_session() as sess:
            converted_scores = sess.run(outputs)
            expected_converted_scores = sess.run(inputs)
            self.assertAllClose(converted_scores, expected_converted_scores)

    def test_build_identity_score_converter_with_logit_scale(self):
        if False:
            return 10
        post_processing_text_proto = '\n      score_converter: IDENTITY\n      logit_scale: 2.0\n    '
        post_processing_config = post_processing_pb2.PostProcessing()
        text_format.Merge(post_processing_text_proto, post_processing_config)
        (_, score_converter) = post_processing_builder.build(post_processing_config)
        self.assertEqual(score_converter.__name__, 'identity_with_logit_scale')
        inputs = tf.constant([1, 1], tf.float32)
        outputs = score_converter(inputs)
        with self.test_session() as sess:
            converted_scores = sess.run(outputs)
            expected_converted_scores = sess.run(tf.constant([0.5, 0.5], tf.float32))
            self.assertAllClose(converted_scores, expected_converted_scores)

    def test_build_sigmoid_score_converter(self):
        if False:
            for i in range(10):
                print('nop')
        post_processing_text_proto = '\n      score_converter: SIGMOID\n    '
        post_processing_config = post_processing_pb2.PostProcessing()
        text_format.Merge(post_processing_text_proto, post_processing_config)
        (_, score_converter) = post_processing_builder.build(post_processing_config)
        self.assertEqual(score_converter.__name__, 'sigmoid_with_logit_scale')

    def test_build_softmax_score_converter(self):
        if False:
            while True:
                i = 10
        post_processing_text_proto = '\n      score_converter: SOFTMAX\n    '
        post_processing_config = post_processing_pb2.PostProcessing()
        text_format.Merge(post_processing_text_proto, post_processing_config)
        (_, score_converter) = post_processing_builder.build(post_processing_config)
        self.assertEqual(score_converter.__name__, 'softmax_with_logit_scale')

    def test_build_softmax_score_converter_with_temperature(self):
        if False:
            print('Hello World!')
        post_processing_text_proto = '\n      score_converter: SOFTMAX\n      logit_scale: 2.0\n    '
        post_processing_config = post_processing_pb2.PostProcessing()
        text_format.Merge(post_processing_text_proto, post_processing_config)
        (_, score_converter) = post_processing_builder.build(post_processing_config)
        self.assertEqual(score_converter.__name__, 'softmax_with_logit_scale')

    def test_build_calibrator_with_nonempty_config(self):
        if False:
            print('Hello World!')
        'Test that identity function used when no calibration_config specified.'
        post_processing_text_proto = '\n      score_converter: SOFTMAX\n      calibration_config {\n        function_approximation {\n          x_y_pairs {\n              x_y_pair {\n                x: 0.0\n                y: 0.5\n              }\n              x_y_pair {\n                x: 1.0\n                y: 0.5\n              }}}}'
        post_processing_config = post_processing_pb2.PostProcessing()
        text_format.Merge(post_processing_text_proto, post_processing_config)
        (_, calibrated_score_conversion_fn) = post_processing_builder.build(post_processing_config)
        self.assertEqual(calibrated_score_conversion_fn.__name__, 'calibrate_with_function_approximation')
        input_scores = tf.constant([1, 1], tf.float32)
        outputs = calibrated_score_conversion_fn(input_scores)
        with self.test_session() as sess:
            calibrated_scores = sess.run(outputs)
            expected_calibrated_scores = sess.run(tf.constant([0.5, 0.5], tf.float32))
            self.assertAllClose(calibrated_scores, expected_calibrated_scores)

    def test_build_temperature_scaling_calibrator(self):
        if False:
            print('Hello World!')
        post_processing_text_proto = '\n      score_converter: SOFTMAX\n      calibration_config {\n        temperature_scaling_calibration {\n          scaler: 2.0\n          }}'
        post_processing_config = post_processing_pb2.PostProcessing()
        text_format.Merge(post_processing_text_proto, post_processing_config)
        (_, calibrated_score_conversion_fn) = post_processing_builder.build(post_processing_config)
        self.assertEqual(calibrated_score_conversion_fn.__name__, 'calibrate_with_temperature_scaling_calibration')
        input_scores = tf.constant([1, 1], tf.float32)
        outputs = calibrated_score_conversion_fn(input_scores)
        with self.test_session() as sess:
            calibrated_scores = sess.run(outputs)
            expected_calibrated_scores = sess.run(tf.constant([0.5, 0.5], tf.float32))
            self.assertAllClose(calibrated_scores, expected_calibrated_scores)
if __name__ == '__main__':
    tf.test.main()