"""Tests for svtcn_loss.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from estimators import svtcn_loss
import tensorflow as tf

class SVTCNLoss(tf.test.TestCase):

    def testSVTCNLoss(self):
        if False:
            i = 10
            return i + 15
        with self.test_session():
            num_data = 64
            num_sequences = 2
            num_data_per_seq = num_data // num_sequences
            feat_dim = 6
            margin = 1.0
            times = np.tile(np.arange(num_data_per_seq, dtype=np.int32), num_sequences)
            times = np.reshape(times, [times.shape[0], 1])
            sequence_ids = np.concatenate([np.ones(num_data_per_seq) * i for i in range(num_sequences)])
            sequence_ids = np.reshape(sequence_ids, [sequence_ids.shape[0], 1])
            pos_radius = 6
            neg_radius = 12
            embedding = np.random.rand(num_data, feat_dim).astype(np.float32)
            in_pos_range = np.less_equal(np.abs(times - times.transpose()), pos_radius)
            in_neg_range = np.greater(np.abs(times - times.transpose()), neg_radius)
            sequence_adjacency = sequence_ids == sequence_ids.T
            sequence_adjacency_not = np.logical_not(sequence_adjacency)
            pdist_matrix = euclidean_distances(embedding, squared=True)
            loss_np = 0.0
            num_positives = 0.0
            for i in range(num_data):
                for j in range(num_data):
                    if in_pos_range[i, j] and i != j and sequence_adjacency[i, j]:
                        num_positives += 1.0
                        pos_distance = pdist_matrix[i][j]
                        neg_distances = []
                        for k in range(num_data):
                            if in_neg_range[i, k] or sequence_adjacency_not[i, k]:
                                neg_distances.append(pdist_matrix[i][k])
                        neg_distances.sort()
                        chosen_neg_distance = neg_distances[0]
                        for l in range(len(neg_distances)):
                            chosen_neg_distance = neg_distances[l]
                            if chosen_neg_distance > pos_distance:
                                break
                        loss_np += np.maximum(0.0, margin - chosen_neg_distance + pos_distance)
            loss_np /= num_positives
            loss_tf = svtcn_loss.singleview_tcn_loss(embeddings=tf.convert_to_tensor(embedding), timesteps=tf.convert_to_tensor(times), pos_radius=pos_radius, neg_radius=neg_radius, margin=margin, sequence_ids=tf.convert_to_tensor(sequence_ids), multiseq=True)
            loss_tf = loss_tf.eval()
            self.assertAllClose(loss_np, loss_tf)
if __name__ == '__main__':
    tf.test.main()