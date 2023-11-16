"""Local feature aggregation extraction.

For more details, please refer to the paper:
"Detect-to-Retrieve: Efficient Regional Aggregation for Image Search",
Proc. CVPR'19 (https://arxiv.org/abs/1812.01584).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from delf import aggregation_config_pb2
_NORM_SQUARED_TOLERANCE = 1e-12
_VLAD = aggregation_config_pb2.AggregationConfig.VLAD
_ASMK = aggregation_config_pb2.AggregationConfig.ASMK
_ASMK_STAR = aggregation_config_pb2.AggregationConfig.ASMK_STAR

class ExtractAggregatedRepresentation(object):
    """Class for extraction of aggregated local feature representation.

  Args:
    sess: TensorFlow session to use.
    aggregation_config: AggregationConfig object defining type of aggregation to
      use.

  Raises:
    ValueError: If aggregation type is invalid.
  """

    def __init__(self, sess, aggregation_config):
        if False:
            return 10
        self._sess = sess
        self._codebook_size = aggregation_config.codebook_size
        self._feature_dimensionality = aggregation_config.feature_dimensionality
        self._aggregation_type = aggregation_config.aggregation_type
        self._feature_batch_size = aggregation_config.feature_batch_size
        self._features = tf.compat.v1.placeholder(tf.float32, [None, None])
        self._num_features_per_region = tf.compat.v1.placeholder(tf.int32, [None])
        codebook = tf.compat.v1.get_variable('codebook', shape=[aggregation_config.codebook_size, aggregation_config.feature_dimensionality])
        tf.compat.v1.train.init_from_checkpoint(aggregation_config.codebook_path, {tf.contrib.factorization.KMeansClustering.CLUSTER_CENTERS_VAR_NAME: codebook})
        if self._aggregation_type == _VLAD:
            self._feature_visual_words = tf.constant(-1, dtype=tf.int32)
            if aggregation_config.use_regional_aggregation:
                self._aggregated_descriptors = self._ComputeRvlad(self._features, self._num_features_per_region, codebook, use_l2_normalization=aggregation_config.use_l2_normalization, num_assignments=aggregation_config.num_assignments)
            else:
                self._aggregated_descriptors = self._ComputeVlad(self._features, codebook, use_l2_normalization=aggregation_config.use_l2_normalization, num_assignments=aggregation_config.num_assignments)
        elif self._aggregation_type == _ASMK or self._aggregation_type == _ASMK_STAR:
            if aggregation_config.use_regional_aggregation:
                (self._aggregated_descriptors, self._feature_visual_words) = self._ComputeRasmk(self._features, self._num_features_per_region, codebook, num_assignments=aggregation_config.num_assignments)
            else:
                (self._aggregated_descriptors, self._feature_visual_words) = self._ComputeAsmk(self._features, codebook, num_assignments=aggregation_config.num_assignments)
        else:
            raise ValueError('Invalid aggregation type: %d' % self._aggregation_type)
        sess.run(tf.compat.v1.global_variables_initializer())

    def Extract(self, features, num_features_per_region=None):
        if False:
            for i in range(10):
                print('nop')
        'Extracts aggregated representation.\n\n    Args:\n      features: [N, D] float numpy array with N local feature descriptors.\n      num_features_per_region: Required only if computing regional aggregated\n        representations, otherwise optional. List of number of features per\n        region, such that sum(num_features_per_region) = N. It indicates which\n        features correspond to each region.\n\n    Returns:\n      aggregated_descriptors: 1-D numpy array.\n      feature_visual_words: Used only for ASMK/ASMK* aggregation type. 1-D\n        numpy array denoting visual words corresponding to the\n        `aggregated_descriptors`.\n\n    Raises:\n      ValueError: If inputs are misconfigured.\n    '
        if num_features_per_region is None:
            num_features_per_region = []
        elif len(num_features_per_region) and sum(num_features_per_region) != features.shape[0]:
            raise ValueError('Incorrect arguments: sum(num_features_per_region) and features.shape[0] are different: %d vs %d' % (sum(num_features_per_region), features.shape[0]))
        (aggregated_descriptors, feature_visual_words) = self._sess.run([self._aggregated_descriptors, self._feature_visual_words], feed_dict={self._features: features, self._num_features_per_region: num_features_per_region})
        if self._aggregation_type == _ASMK_STAR:
            reshaped_aggregated_descriptors = np.reshape(aggregated_descriptors, [-1, self._feature_dimensionality])
            packed_descriptors = np.packbits(reshaped_aggregated_descriptors > 0, axis=1)
            aggregated_descriptors = np.reshape(packed_descriptors, [-1])
        return (aggregated_descriptors, feature_visual_words)

    def _ComputeVlad(self, features, codebook, use_l2_normalization=True, num_assignments=1):
        if False:
            return 10
        'Compute VLAD representation.\n\n    Args:\n      features: [N, D] float tensor.\n      codebook: [K, D] float tensor.\n      use_l2_normalization: If False, does not L2-normalize after aggregation.\n      num_assignments: Number of visual words to assign a feature to.\n\n    Returns:\n      vlad: [K*D] float tensor.\n    '

        def _ComputeVladEmptyFeatures():
            if False:
                for i in range(10):
                    print('nop')
            'Computes VLAD if `features` is empty.\n\n      Returns:\n        [K*D] all-zeros tensor.\n      '
            return tf.zeros([self._codebook_size * self._feature_dimensionality], dtype=tf.float32)

        def _ComputeVladNonEmptyFeatures():
            if False:
                for i in range(10):
                    print('nop')
            'Computes VLAD if `features` is not empty.\n\n      Returns:\n        [K*D] tensor with VLAD descriptor.\n      '
            num_features = tf.shape(features)[0]
            if self._feature_batch_size <= 0:
                actual_batch_size = num_features
            else:
                actual_batch_size = self._feature_batch_size

            def _BatchNearestVisualWords(ind, selected_visual_words):
                if False:
                    for i in range(10):
                        print('nop')
                'Compute nearest neighbor visual words for a batch of features.\n\n        Args:\n          ind: Integer index denoting feature.\n          selected_visual_words: Partial set of visual words.\n\n        Returns:\n          output_ind: Next index.\n          output_selected_visual_words: Updated set of visual words, including\n            the visual words for the new batch.\n        '
                batch_size_to_use = tf.cond(tf.greater(ind + actual_batch_size, num_features), true_fn=lambda : num_features - ind, false_fn=lambda : actual_batch_size)
                tiled_features = tf.reshape(tf.tile(tf.slice(features, [ind, 0], [batch_size_to_use, self._feature_dimensionality]), [1, self._codebook_size]), [-1, self._feature_dimensionality])
                tiled_codebook = tf.reshape(tf.tile(tf.reshape(codebook, [1, -1]), [batch_size_to_use, 1]), [-1, self._feature_dimensionality])
                squared_distances = tf.reshape(tf.reduce_sum(tf.math.squared_difference(tiled_features, tiled_codebook), axis=1), [batch_size_to_use, self._codebook_size])
                nearest_visual_words = tf.argsort(squared_distances)
                batch_selected_visual_words = tf.slice(nearest_visual_words, [0, 0], [batch_size_to_use, num_assignments])
                selected_visual_words = tf.concat([selected_visual_words, batch_selected_visual_words], axis=0)
                return (ind + batch_size_to_use, selected_visual_words)
            ind_batch = tf.constant(0, dtype=tf.int32)
            keep_going = lambda j, selected_visual_words: tf.less(j, num_features)
            selected_visual_words = tf.zeros([0, num_assignments], dtype=tf.int32)
            (_, selected_visual_words) = tf.while_loop(cond=keep_going, body=_BatchNearestVisualWords, loop_vars=[ind_batch, selected_visual_words], shape_invariants=[ind_batch.get_shape(), tf.TensorShape([None, num_assignments])], parallel_iterations=1, back_prop=False)

            def _ConstructVladFromAssignments(ind, vlad):
                if False:
                    for i in range(10):
                        print('nop')
                'Add contributions of a feature to a VLAD descriptor.\n\n        Args:\n          ind: Integer index denoting feature.\n          vlad: Partial VLAD descriptor.\n\n        Returns:\n          output_ind: Next index (ie, ind+1).\n          output_vlad: VLAD descriptor updated to take into account contribution\n            from ind-th feature.\n        '
                return (ind + 1, tf.compat.v1.tensor_scatter_add(vlad, tf.expand_dims(selected_visual_words[ind], axis=1), tf.tile(tf.expand_dims(features[ind], axis=0), [num_assignments, 1]) - tf.gather(codebook, selected_visual_words[ind])))
            ind_vlad = tf.constant(0, dtype=tf.int32)
            keep_going = lambda j, vlad: tf.less(j, num_features)
            vlad = tf.zeros([self._codebook_size, self._feature_dimensionality], dtype=tf.float32)
            (_, vlad) = tf.while_loop(cond=keep_going, body=_ConstructVladFromAssignments, loop_vars=[ind_vlad, vlad], back_prop=False)
            vlad = tf.reshape(vlad, [self._codebook_size * self._feature_dimensionality])
            if use_l2_normalization:
                vlad = tf.math.l2_normalize(vlad, epsilon=_NORM_SQUARED_TOLERANCE)
            return vlad
        return tf.cond(tf.greater(tf.size(features), 0), true_fn=_ComputeVladNonEmptyFeatures, false_fn=_ComputeVladEmptyFeatures)

    def _ComputeRvlad(self, features, num_features_per_region, codebook, use_l2_normalization=False, num_assignments=1):
        if False:
            print('Hello World!')
        'Compute R-VLAD representation.\n\n    Args:\n      features: [N, D] float tensor.\n      num_features_per_region: [R] int tensor. Contains number of features per\n        region, such that sum(num_features_per_region) = N. It indicates which\n        features correspond to each region.\n      codebook: [K, D] float tensor.\n      use_l2_normalization: If True, performs L2-normalization after regional\n        aggregation; if False (default), performs componentwise division by R\n        after regional aggregation.\n      num_assignments: Number of visual words to assign a feature to.\n\n    Returns:\n      rvlad: [K*D] float tensor.\n    '

        def _ComputeRvladEmptyRegions():
            if False:
                for i in range(10):
                    print('nop')
            'Computes R-VLAD if `num_features_per_region` is empty.\n\n      Returns:\n        [K*D] all-zeros tensor.\n      '
            return tf.zeros([self._codebook_size * self._feature_dimensionality], dtype=tf.float32)

        def _ComputeRvladNonEmptyRegions():
            if False:
                print('Hello World!')
            'Computes R-VLAD if `num_features_per_region` is not empty.\n\n      Returns:\n        [K*D] tensor with R-VLAD descriptor.\n      '

            def _ConstructRvladFromVlad(ind, rvlad):
                if False:
                    for i in range(10):
                        print('nop')
                'Add contributions from different regions into R-VLAD.\n\n        Args:\n          ind: Integer index denoting region.\n          rvlad: Partial R-VLAD descriptor.\n\n        Returns:\n          output_ind: Next index (ie, ind+1).\n          output_rvlad: R-VLAD descriptor updated to take into account\n            contribution from ind-th region.\n        '
                return (ind + 1, rvlad + self._ComputeVlad(tf.slice(features, [tf.reduce_sum(num_features_per_region[:ind]), 0], [num_features_per_region[ind], self._feature_dimensionality]), codebook, num_assignments=num_assignments))
            i = tf.constant(0, dtype=tf.int32)
            num_regions = tf.shape(num_features_per_region)[0]
            keep_going = lambda j, rvlad: tf.less(j, num_regions)
            rvlad = tf.zeros([self._codebook_size * self._feature_dimensionality], dtype=tf.float32)
            (_, rvlad) = tf.while_loop(cond=keep_going, body=_ConstructRvladFromVlad, loop_vars=[i, rvlad], back_prop=False, parallel_iterations=1)
            if use_l2_normalization:
                rvlad = tf.math.l2_normalize(rvlad, epsilon=_NORM_SQUARED_TOLERANCE)
            else:
                rvlad /= tf.cast(num_regions, dtype=tf.float32)
            return rvlad
        return tf.cond(tf.greater(tf.size(num_features_per_region), 0), true_fn=_ComputeRvladNonEmptyRegions, false_fn=_ComputeRvladEmptyRegions)

    def _PerCentroidNormalization(self, unnormalized_vector):
        if False:
            print('Hello World!')
        'Perform per-centroid normalization.\n\n    Args:\n      unnormalized_vector: [KxD] float tensor.\n\n    Returns:\n      per_centroid_normalized_vector: [KxD] float tensor, with normalized\n        aggregated residuals. Some residuals may be all-zero.\n      visual_words: Int tensor containing indices of visual words which are\n        present for the set of features.\n    '
        unnormalized_vector = tf.reshape(unnormalized_vector, [self._codebook_size, self._feature_dimensionality])
        per_centroid_norms = tf.norm(unnormalized_vector, axis=1)
        visual_words = tf.reshape(tf.where(tf.greater(per_centroid_norms, tf.sqrt(_NORM_SQUARED_TOLERANCE))), [-1])
        per_centroid_normalized_vector = tf.math.l2_normalize(unnormalized_vector, axis=1, epsilon=_NORM_SQUARED_TOLERANCE)
        return (per_centroid_normalized_vector, visual_words)

    def _ComputeAsmk(self, features, codebook, num_assignments=1):
        if False:
            while True:
                i = 10
        'Compute ASMK representation.\n\n    Args:\n      features: [N, D] float tensor.\n      codebook: [K, D] float tensor.\n      num_assignments: Number of visual words to assign a feature to.\n\n    Returns:\n      normalized_residuals: 1-dimensional float tensor with concatenated\n        residuals which are non-zero. Note that the dimensionality is\n        input-dependent.\n      visual_words: 1-dimensional int tensor of sorted visual word ids.\n        Dimensionality is shape(normalized_residuals)[0] / D.\n    '
        unnormalized_vlad = self._ComputeVlad(features, codebook, use_l2_normalization=False, num_assignments=num_assignments)
        (per_centroid_normalized_vlad, visual_words) = self._PerCentroidNormalization(unnormalized_vlad)
        normalized_residuals = tf.reshape(tf.gather(per_centroid_normalized_vlad, visual_words), [tf.shape(visual_words)[0] * self._feature_dimensionality])
        return (normalized_residuals, visual_words)

    def _ComputeRasmk(self, features, num_features_per_region, codebook, num_assignments=1):
        if False:
            i = 10
            return i + 15
        'Compute R-ASMK representation.\n\n    Args:\n      features: [N, D] float tensor.\n      num_features_per_region: [R] int tensor. Contains number of features per\n        region, such that sum(num_features_per_region) = N. It indicates which\n        features correspond to each region.\n      codebook: [K, D] float tensor.\n      num_assignments: Number of visual words to assign a feature to.\n\n    Returns:\n      normalized_residuals: 1-dimensional float tensor with concatenated\n        residuals which are non-zero. Note that the dimensionality is\n        input-dependent.\n      visual_words: 1-dimensional int tensor of sorted visual word ids.\n        Dimensionality is shape(normalized_residuals)[0] / D.\n    '
        unnormalized_rvlad = self._ComputeRvlad(features, num_features_per_region, codebook, use_l2_normalization=False, num_assignments=num_assignments)
        (per_centroid_normalized_rvlad, visual_words) = self._PerCentroidNormalization(unnormalized_rvlad)
        normalized_residuals = tf.reshape(tf.gather(per_centroid_normalized_rvlad, visual_words), [tf.shape(visual_words)[0] * self._feature_dimensionality])
        return (normalized_residuals, visual_words)