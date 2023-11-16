"""Tests for DELF feature aggregation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from delf import aggregation_config_pb2
from delf import feature_aggregation_extractor

class FeatureAggregationTest(tf.test.TestCase):

    def _CreateCodebook(self, checkpoint_path):
        if False:
            i = 10
            return i + 15
        'Creates codebook used in tests.\n\n    Args:\n      checkpoint_path: Directory where codebook is saved to.\n    '
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            codebook = tf.Variable([[0.5, 0.5], [0.0, 0.0], [1.0, 0.0], [-0.5, -0.5], [0.0, 1.0]], name='clusters')
            saver = tf.compat.v1.train.Saver([codebook])
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.save(sess, checkpoint_path)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._codebook_path = os.path.join(tf.compat.v1.test.get_temp_dir(), 'test_codebook')
        self._CreateCodebook(self._codebook_path)

    def testComputeNormalizedVladWorks(self):
        if False:
            for i in range(10):
                print('nop')
        features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0]], dtype=float)
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
        config.use_l2_normalization = True
        config.codebook_path = self._codebook_path
        config.num_assignments = 1
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            (vlad, extra_output) = extractor.Extract(features)
        exp_vlad = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.316228, 0.316228, 0.632456, 0.632456]
        exp_extra_output = -1
        self.assertAllClose(vlad, exp_vlad)
        self.assertAllEqual(extra_output, exp_extra_output)

    def testComputeNormalizedVladWithBatchingWorks(self):
        if False:
            print('Hello World!')
        features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0]], dtype=float)
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
        config.use_l2_normalization = True
        config.codebook_path = self._codebook_path
        config.num_assignments = 1
        config.feature_batch_size = 2
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            (vlad, extra_output) = extractor.Extract(features)
        exp_vlad = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.316228, 0.316228, 0.632456, 0.632456]
        exp_extra_output = -1
        self.assertAllClose(vlad, exp_vlad)
        self.assertAllEqual(extra_output, exp_extra_output)

    def testComputeUnnormalizedVladWorks(self):
        if False:
            return 10
        features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0]], dtype=float)
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
        config.use_l2_normalization = False
        config.codebook_path = self._codebook_path
        config.num_assignments = 1
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            (vlad, extra_output) = extractor.Extract(features)
        exp_vlad = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.5, 1.0, 1.0]
        exp_extra_output = -1
        self.assertAllEqual(vlad, exp_vlad)
        self.assertAllEqual(extra_output, exp_extra_output)

    def testComputeUnnormalizedVladMultipleAssignmentWorks(self):
        if False:
            print('Hello World!')
        features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0]], dtype=float)
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
        config.use_l2_normalization = False
        config.codebook_path = self._codebook_path
        config.num_assignments = 3
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            (vlad, extra_output) = extractor.Extract(features)
        exp_vlad = [1.0, 1.0, 0.0, 0.0, 0.0, 2.0, -0.5, 0.5, 0.0, 0.0]
        exp_extra_output = -1
        self.assertAllEqual(vlad, exp_vlad)
        self.assertAllEqual(extra_output, exp_extra_output)

    def testComputeVladEmptyFeaturesWorks(self):
        if False:
            print('Hello World!')
        features = np.array([[]])
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
        config.codebook_path = self._codebook_path
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            (vlad, extra_output) = extractor.Extract(features)
        exp_vlad = np.zeros([10], dtype=float)
        exp_extra_output = -1
        self.assertAllEqual(vlad, exp_vlad)
        self.assertAllEqual(extra_output, exp_extra_output)

    def testComputeUnnormalizedRvladWorks(self):
        if False:
            print('Hello World!')
        features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0], [0.0, 2.0]], dtype=float)
        num_features_per_region = np.array([3, 1])
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
        config.use_l2_normalization = False
        config.codebook_path = self._codebook_path
        config.num_assignments = 1
        config.use_regional_aggregation = True
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            (rvlad, extra_output) = extractor.Extract(features, num_features_per_region)
        exp_rvlad = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.158114, 0.158114, 0.316228, 0.816228]
        exp_extra_output = -1
        self.assertAllClose(rvlad, exp_rvlad)
        self.assertAllEqual(extra_output, exp_extra_output)

    def testComputeNormalizedRvladWorks(self):
        if False:
            return 10
        features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0], [0.0, 2.0]], dtype=float)
        num_features_per_region = np.array([3, 1])
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
        config.use_l2_normalization = True
        config.codebook_path = self._codebook_path
        config.num_assignments = 1
        config.use_regional_aggregation = True
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            (rvlad, extra_output) = extractor.Extract(features, num_features_per_region)
        exp_rvlad = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.175011, 0.175011, 0.350021, 0.903453]
        exp_extra_output = -1
        self.assertAllClose(rvlad, exp_rvlad)
        self.assertAllEqual(extra_output, exp_extra_output)

    def testComputeRvladEmptyRegionsWorks(self):
        if False:
            for i in range(10):
                print('nop')
        features = np.array([[]])
        num_features_per_region = np.array([])
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
        config.codebook_path = self._codebook_path
        config.use_regional_aggregation = True
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            (rvlad, extra_output) = extractor.Extract(features, num_features_per_region)
        exp_rvlad = np.zeros([10], dtype=float)
        exp_extra_output = -1
        self.assertAllEqual(rvlad, exp_rvlad)
        self.assertAllEqual(extra_output, exp_extra_output)

    def testComputeUnnormalizedRvladSomeEmptyRegionsWorks(self):
        if False:
            return 10
        features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0], [0.0, 2.0]], dtype=float)
        num_features_per_region = np.array([0, 3, 0, 1])
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
        config.use_l2_normalization = False
        config.codebook_path = self._codebook_path
        config.num_assignments = 1
        config.use_regional_aggregation = True
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            (rvlad, extra_output) = extractor.Extract(features, num_features_per_region)
        exp_rvlad = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.079057, 0.079057, 0.158114, 0.408114]
        exp_extra_output = -1
        self.assertAllClose(rvlad, exp_rvlad)
        self.assertAllEqual(extra_output, exp_extra_output)

    def testComputeNormalizedRvladSomeEmptyRegionsWorks(self):
        if False:
            for i in range(10):
                print('nop')
        features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0], [0.0, 2.0]], dtype=float)
        num_features_per_region = np.array([0, 3, 0, 1])
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
        config.use_l2_normalization = True
        config.codebook_path = self._codebook_path
        config.num_assignments = 1
        config.use_regional_aggregation = True
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            (rvlad, extra_output) = extractor.Extract(features, num_features_per_region)
        exp_rvlad = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.175011, 0.175011, 0.350021, 0.903453]
        exp_extra_output = -1
        self.assertAllClose(rvlad, exp_rvlad)
        self.assertAllEqual(extra_output, exp_extra_output)

    def testComputeRvladMisconfiguredFeatures(self):
        if False:
            for i in range(10):
                print('nop')
        features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0], [0.0, 2.0]], dtype=float)
        num_features_per_region = np.array([3, 2])
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
        config.codebook_path = self._codebook_path
        config.use_regional_aggregation = True
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            with self.assertRaisesRegex(ValueError, 'Incorrect arguments: sum\\(num_features_per_region\\) and features.shape\\[0\\] are different'):
                extractor.Extract(features, num_features_per_region)

    def testComputeAsmkWorks(self):
        if False:
            print('Hello World!')
        features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0]], dtype=float)
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK
        config.codebook_path = self._codebook_path
        config.num_assignments = 1
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            (asmk, visual_words) = extractor.Extract(features)
        exp_asmk = [-0.707107, 0.707107, 0.707107, 0.707107]
        exp_visual_words = [3, 4]
        self.assertAllClose(asmk, exp_asmk)
        self.assertAllEqual(visual_words, exp_visual_words)

    def testComputeAsmkStarWorks(self):
        if False:
            print('Hello World!')
        features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0]], dtype=float)
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK_STAR
        config.codebook_path = self._codebook_path
        config.num_assignments = 1
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            (asmk_star, visual_words) = extractor.Extract(features)
        exp_asmk_star = [64, 192]
        exp_visual_words = [3, 4]
        self.assertAllEqual(asmk_star, exp_asmk_star)
        self.assertAllEqual(visual_words, exp_visual_words)

    def testComputeAsmkMultipleAssignmentWorks(self):
        if False:
            while True:
                i = 10
        features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0]], dtype=float)
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK
        config.codebook_path = self._codebook_path
        config.num_assignments = 3
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            (asmk, visual_words) = extractor.Extract(features)
        exp_asmk = [0.707107, 0.707107, 0.0, 1.0, -0.707107, 0.707107]
        exp_visual_words = [0, 2, 3]
        self.assertAllClose(asmk, exp_asmk)
        self.assertAllEqual(visual_words, exp_visual_words)

    def testComputeRasmkWorks(self):
        if False:
            for i in range(10):
                print('nop')
        features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0], [0.0, 2.0]], dtype=float)
        num_features_per_region = np.array([3, 1])
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK
        config.codebook_path = self._codebook_path
        config.num_assignments = 1
        config.use_regional_aggregation = True
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            (rasmk, visual_words) = extractor.Extract(features, num_features_per_region)
        exp_rasmk = [-0.707107, 0.707107, 0.361261, 0.932465]
        exp_visual_words = [3, 4]
        self.assertAllClose(rasmk, exp_rasmk)
        self.assertAllEqual(visual_words, exp_visual_words)

    def testComputeRasmkStarWorks(self):
        if False:
            i = 10
            return i + 15
        features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0], [0.0, 2.0]], dtype=float)
        num_features_per_region = np.array([3, 1])
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK_STAR
        config.codebook_path = self._codebook_path
        config.num_assignments = 1
        config.use_regional_aggregation = True
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
            (rasmk_star, visual_words) = extractor.Extract(features, num_features_per_region)
        exp_rasmk_star = [64, 192]
        exp_visual_words = [3, 4]
        self.assertAllEqual(rasmk_star, exp_rasmk_star)
        self.assertAllEqual(visual_words, exp_visual_words)

    def testComputeUnknownAggregation(self):
        if False:
            print('Hello World!')
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = 0
        config.codebook_path = self._codebook_path
        config.use_regional_aggregation = True
        with tf.Graph().as_default() as g, self.session(graph=g) as sess:
            with self.assertRaisesRegex(ValueError, 'Invalid aggregation type'):
                feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)
if __name__ == '__main__':
    tf.test.main()