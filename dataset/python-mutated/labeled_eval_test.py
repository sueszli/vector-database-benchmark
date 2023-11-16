"""Tests for tcn.labeled_eval."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import labeled_eval
import tensorflow as tf

class LabeledEvalTest(tf.test.TestCase):

    def testNearestCrossSequenceNeighbors(self):
        if False:
            for i in range(10):
                print('nop')
        num_data = 64
        embedding_size = 4
        num_tasks = 8
        n_neighbors = 2
        data = np.random.randn(num_data, embedding_size)
        tasks = np.repeat(range(num_tasks), num_data // num_tasks)
        indices = labeled_eval.nearest_cross_sequence_neighbors(data, tasks, n_neighbors=n_neighbors)
        repeated_tasks = np.tile(np.reshape(tasks, (num_data, 1)), n_neighbors)
        self.assertTrue(np.all(np.not_equal(repeated_tasks, tasks[indices])))

    def testPerfectCrossSequenceRecall(self):
        if False:
            while True:
                i = 10
        embeddings = np.random.randn(10, 2)
        embeddings[5:, :] = 1e-05 + embeddings[:5, :]
        tasks = np.repeat([0, 1], 5)
        labels = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        k_list = [1, 2]
        indices = labeled_eval.nearest_cross_sequence_neighbors(embeddings, tasks, n_neighbors=max(k_list))
        retrieved_labels = labels[indices]
        recall_list = labeled_eval.compute_cross_sequence_recall_at_k(retrieved_labels=retrieved_labels, labels=labels, k_list=k_list)
        self.assertTrue(np.allclose(np.array(recall_list), np.array([1.0, 1.0])))

    def testRelativeRecall(self):
        if False:
            while True:
                i = 10
        num_data = 100
        num_tasks = 10
        embeddings = np.random.randn(100, 5)
        tasks = np.repeat(range(num_tasks), num_data // num_tasks)
        labels = np.random.randint(0, 5, 100)
        k_list = [1, 2, 4, 8, 16, 32, 64]
        indices = labeled_eval.nearest_cross_sequence_neighbors(embeddings, tasks, n_neighbors=max(k_list))
        retrieved_labels = labels[indices]
        recall_list = labeled_eval.compute_cross_sequence_recall_at_k(retrieved_labels=retrieved_labels, labels=labels, k_list=k_list)
        recall_list_sorted = sorted(recall_list)
        self.assertTrue(np.allclose(np.array(recall_list), np.array(recall_list_sorted)))
if __name__ == '__main__':
    tf.test.main()