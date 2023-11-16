"""Bipartite matcher implementation."""
import tensorflow as tf
from tensorflow.contrib.image.python.ops import image_ops
from object_detection.core import matcher

class GreedyBipartiteMatcher(matcher.Matcher):
    """Wraps a Tensorflow greedy bipartite matcher."""

    def __init__(self, use_matmul_gather=False):
        if False:
            print('Hello World!')
        'Constructs a Matcher.\n\n    Args:\n      use_matmul_gather: Force constructed match objects to use matrix\n        multiplication based gather instead of standard tf.gather.\n        (Default: False).\n    '
        super(GreedyBipartiteMatcher, self).__init__(use_matmul_gather=use_matmul_gather)

    def _match(self, similarity_matrix, valid_rows):
        if False:
            for i in range(10):
                print('nop')
        'Bipartite matches a collection rows and columns. A greedy bi-partite.\n\n    TODO(rathodv): Add num_valid_columns options to match only that many columns\n    with all the rows.\n\n    Args:\n      similarity_matrix: Float tensor of shape [N, M] with pairwise similarity\n        where higher values mean more similar.\n      valid_rows: A boolean tensor of shape [N] indicating the rows that are\n        valid.\n\n    Returns:\n      match_results: int32 tensor of shape [M] with match_results[i]=-1\n        meaning that column i is not matched and otherwise that it is matched to\n        row match_results[i].\n    '
        valid_row_sim_matrix = tf.gather(similarity_matrix, tf.squeeze(tf.where(valid_rows), axis=-1))
        invalid_row_sim_matrix = tf.gather(similarity_matrix, tf.squeeze(tf.where(tf.logical_not(valid_rows)), axis=-1))
        similarity_matrix = tf.concat([valid_row_sim_matrix, invalid_row_sim_matrix], axis=0)
        distance_matrix = -1 * similarity_matrix
        num_valid_rows = tf.reduce_sum(tf.cast(valid_rows, dtype=tf.float32))
        (_, match_results) = image_ops.bipartite_match(distance_matrix, num_valid_rows=num_valid_rows)
        match_results = tf.reshape(match_results, [-1])
        match_results = tf.cast(match_results, tf.int32)
        return match_results