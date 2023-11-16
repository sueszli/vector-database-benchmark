"""Tests for data_providers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import data_providers
import tensorflow as tf

class DataTest(tf.test.TestCase):

    def testMVTripletIndices(self):
        if False:
            print('Hello World!')
        'Ensures anchor/pos indices for a TCN batch are valid.'
        tf.set_random_seed(0)
        window = 580
        batch_size = 36
        num_pairs = batch_size // 2
        num_views = 2
        seq_len = 600
        (_, a_view_indices, p_view_indices) = data_providers.get_tcn_anchor_pos_indices(seq_len, num_views, num_pairs, window)
        with self.test_session() as sess:
            (np_a_view_indices, np_p_view_indices) = sess.run([a_view_indices, p_view_indices])
            np.testing.assert_equal(np.any(np.not_equal(np_a_view_indices, np_p_view_indices)), True)
            view_set = set(range(num_views))
            self.assertTrue(set(np_a_view_indices).issubset(view_set))
            self.assertTrue(set(np_p_view_indices).issubset(view_set))

    def testSVTripletIndices(self):
        if False:
            print('Hello World!')
        'Ensures time indices for a SV triplet batch are valid.'
        seq_len = 600
        batch_size = 36
        num_views = 2
        (time_indices, _) = data_providers.get_svtcn_indices(seq_len, batch_size, num_views)
        with self.test_session() as sess:
            np_time_indices = sess.run(time_indices)
            first = np_time_indices[0]
            last = np_time_indices[-1]
            self.assertTrue(np.array_equal(np_time_indices, range(first, last + 1)))
if __name__ == '__main__':
    tf.test.main()