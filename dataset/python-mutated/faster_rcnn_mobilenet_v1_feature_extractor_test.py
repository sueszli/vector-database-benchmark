"""Tests for faster_rcnn_mobilenet_v1_feature_extractor."""
import numpy as np
import tensorflow as tf
from object_detection.models import faster_rcnn_mobilenet_v1_feature_extractor as faster_rcnn_mobilenet_v1

class FasterRcnnMobilenetV1FeatureExtractorTest(tf.test.TestCase):

    def _build_feature_extractor(self, first_stage_features_stride):
        if False:
            print('Hello World!')
        return faster_rcnn_mobilenet_v1.FasterRCNNMobilenetV1FeatureExtractor(is_training=False, first_stage_features_stride=first_stage_features_stride, batch_norm_trainable=False, reuse_weights=None, weight_decay=0.0)

    def test_extract_proposal_features_returns_expected_size(self):
        if False:
            i = 10
            return i + 15
        feature_extractor = self._build_feature_extractor(first_stage_features_stride=16)
        preprocessed_inputs = tf.random_uniform([4, 224, 224, 3], maxval=255, dtype=tf.float32)
        (rpn_feature_map, _) = feature_extractor.extract_proposal_features(preprocessed_inputs, scope='TestScope')
        features_shape = tf.shape(rpn_feature_map)
        init_op = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init_op)
            features_shape_out = sess.run(features_shape)
            self.assertAllEqual(features_shape_out, [4, 14, 14, 512])

    def test_extract_proposal_features_stride_eight(self):
        if False:
            i = 10
            return i + 15
        feature_extractor = self._build_feature_extractor(first_stage_features_stride=8)
        preprocessed_inputs = tf.random_uniform([4, 224, 224, 3], maxval=255, dtype=tf.float32)
        (rpn_feature_map, _) = feature_extractor.extract_proposal_features(preprocessed_inputs, scope='TestScope')
        features_shape = tf.shape(rpn_feature_map)
        init_op = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init_op)
            features_shape_out = sess.run(features_shape)
            self.assertAllEqual(features_shape_out, [4, 14, 14, 512])

    def test_extract_proposal_features_half_size_input(self):
        if False:
            i = 10
            return i + 15
        feature_extractor = self._build_feature_extractor(first_stage_features_stride=16)
        preprocessed_inputs = tf.random_uniform([1, 112, 112, 3], maxval=255, dtype=tf.float32)
        (rpn_feature_map, _) = feature_extractor.extract_proposal_features(preprocessed_inputs, scope='TestScope')
        features_shape = tf.shape(rpn_feature_map)
        init_op = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init_op)
            features_shape_out = sess.run(features_shape)
            self.assertAllEqual(features_shape_out, [1, 7, 7, 512])

    def test_extract_proposal_features_dies_on_invalid_stride(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            self._build_feature_extractor(first_stage_features_stride=99)

    def test_extract_proposal_features_dies_on_very_small_images(self):
        if False:
            for i in range(10):
                print('nop')
        feature_extractor = self._build_feature_extractor(first_stage_features_stride=16)
        preprocessed_inputs = tf.placeholder(tf.float32, (4, None, None, 3))
        (rpn_feature_map, _) = feature_extractor.extract_proposal_features(preprocessed_inputs, scope='TestScope')
        features_shape = tf.shape(rpn_feature_map)
        init_op = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init_op)
            with self.assertRaises(tf.errors.InvalidArgumentError):
                sess.run(features_shape, feed_dict={preprocessed_inputs: np.random.rand(4, 32, 32, 3)})

    def test_extract_proposal_features_dies_with_incorrect_rank_inputs(self):
        if False:
            return 10
        feature_extractor = self._build_feature_extractor(first_stage_features_stride=16)
        preprocessed_inputs = tf.random_uniform([224, 224, 3], maxval=255, dtype=tf.float32)
        with self.assertRaises(ValueError):
            feature_extractor.extract_proposal_features(preprocessed_inputs, scope='TestScope')

    def test_extract_box_classifier_features_returns_expected_size(self):
        if False:
            while True:
                i = 10
        feature_extractor = self._build_feature_extractor(first_stage_features_stride=16)
        proposal_feature_maps = tf.random_uniform([3, 14, 14, 576], maxval=255, dtype=tf.float32)
        proposal_classifier_features = feature_extractor.extract_box_classifier_features(proposal_feature_maps, scope='TestScope')
        features_shape = tf.shape(proposal_classifier_features)
        init_op = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init_op)
            features_shape_out = sess.run(features_shape)
            self.assertAllEqual(features_shape_out, [3, 7, 7, 1024])
if __name__ == '__main__':
    tf.test.main()