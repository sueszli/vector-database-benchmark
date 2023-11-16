"""Tests for DELF feature aggregation similarity."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from delf import aggregation_config_pb2
from delf import feature_aggregation_similarity

class FeatureAggregationSimilarityTest(tf.test.TestCase):

    def testComputeVladSimilarityWorks(self):
        if False:
            while True:
                i = 10
        vlad_1 = np.array([0, 1, 2, 3, 4])
        vlad_2 = np.array([5, 6, 7, 8, 9])
        config = aggregation_config_pb2.AggregationConfig()
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
        similarity_computer = feature_aggregation_similarity.SimilarityAggregatedRepresentation(config)
        similarity = similarity_computer.ComputeSimilarity(vlad_1, vlad_2)
        exp_similarity = 80
        self.assertAllEqual(similarity, exp_similarity)

    def testComputeAsmkSimilarityWorks(self):
        if False:
            i = 10
            return i + 15
        aggregated_descriptors_1 = np.array([0.0, 0.0, -0.707107, -0.707107, 0.5, 0.866025, 0.816497, 0.57735, 1.0, 0.0])
        visual_words_1 = np.array([0, 1, 2, 3, 4])
        aggregated_descriptors_2 = np.array([0.0, 1.0, 1.0, 0.0, 0.707107, 0.707107])
        visual_words_2 = np.array([1, 2, 4])
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK
        config.use_l2_normalization = True
        similarity_computer = feature_aggregation_similarity.SimilarityAggregatedRepresentation(config)
        similarity = similarity_computer.ComputeSimilarity(aggregated_descriptors_1, aggregated_descriptors_2, visual_words_1, visual_words_2)
        exp_similarity = 0.123562
        self.assertAllClose(similarity, exp_similarity)

    def testComputeAsmkSimilarityNoNormalizationWorks(self):
        if False:
            for i in range(10):
                print('nop')
        aggregated_descriptors_1 = np.array([0.0, 0.0, -0.707107, -0.707107, 0.5, 0.866025, 0.816497, 0.57735, 1.0, 0.0])
        visual_words_1 = np.array([0, 1, 2, 3, 4])
        aggregated_descriptors_2 = np.array([0.0, 1.0, 1.0, 0.0, 0.707107, 0.707107])
        visual_words_2 = np.array([1, 2, 4])
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK
        config.use_l2_normalization = False
        similarity_computer = feature_aggregation_similarity.SimilarityAggregatedRepresentation(config)
        similarity = similarity_computer.ComputeSimilarity(aggregated_descriptors_1, aggregated_descriptors_2, visual_words_1, visual_words_2)
        exp_similarity = 0.478554
        self.assertAllClose(similarity, exp_similarity)

    def testComputeAsmkStarSimilarityWorks(self):
        if False:
            print('Hello World!')
        aggregated_descriptors_1 = np.array([0, 0, 3, 3, 3], dtype='uint8')
        visual_words_1 = np.array([0, 1, 2, 3, 4])
        aggregated_descriptors_2 = np.array([1, 2, 3], dtype='uint8')
        visual_words_2 = np.array([1, 2, 4])
        config = aggregation_config_pb2.AggregationConfig()
        config.codebook_size = 5
        config.feature_dimensionality = 2
        config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK_STAR
        config.use_l2_normalization = True
        similarity_computer = feature_aggregation_similarity.SimilarityAggregatedRepresentation(config)
        similarity = similarity_computer.ComputeSimilarity(aggregated_descriptors_1, aggregated_descriptors_2, visual_words_1, visual_words_2)
        exp_similarity = 0.258199
        self.assertAllClose(similarity, exp_similarity)
if __name__ == '__main__':
    tf.test.main()