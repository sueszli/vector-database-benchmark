"""Tests for object_detection.core.matcher."""
import numpy as np
import tensorflow as tf
from object_detection.core import matcher

class MatchTest(tf.test.TestCase):

    def test_get_correct_matched_columnIndices(self):
        if False:
            while True:
                i = 10
        match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
        match = matcher.Match(match_results)
        expected_column_indices = [0, 1, 3, 5]
        matched_column_indices = match.matched_column_indices()
        self.assertEqual(matched_column_indices.dtype, tf.int32)
        with self.test_session() as sess:
            matched_column_indices = sess.run(matched_column_indices)
            self.assertAllEqual(matched_column_indices, expected_column_indices)

    def test_get_correct_counts(self):
        if False:
            while True:
                i = 10
        match_results = tf.constant([3, 1, -1, 0, -1, 1, -2])
        match = matcher.Match(match_results)
        exp_num_matched_columns = 4
        exp_num_unmatched_columns = 2
        exp_num_ignored_columns = 1
        exp_num_matched_rows = 3
        num_matched_columns = match.num_matched_columns()
        num_unmatched_columns = match.num_unmatched_columns()
        num_ignored_columns = match.num_ignored_columns()
        num_matched_rows = match.num_matched_rows()
        self.assertEqual(num_matched_columns.dtype, tf.int32)
        self.assertEqual(num_unmatched_columns.dtype, tf.int32)
        self.assertEqual(num_ignored_columns.dtype, tf.int32)
        self.assertEqual(num_matched_rows.dtype, tf.int32)
        with self.test_session() as sess:
            (num_matched_columns_out, num_unmatched_columns_out, num_ignored_columns_out, num_matched_rows_out) = sess.run([num_matched_columns, num_unmatched_columns, num_ignored_columns, num_matched_rows])
            self.assertAllEqual(num_matched_columns_out, exp_num_matched_columns)
            self.assertAllEqual(num_unmatched_columns_out, exp_num_unmatched_columns)
            self.assertAllEqual(num_ignored_columns_out, exp_num_ignored_columns)
            self.assertAllEqual(num_matched_rows_out, exp_num_matched_rows)

    def testGetCorrectUnmatchedColumnIndices(self):
        if False:
            for i in range(10):
                print('nop')
        match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
        match = matcher.Match(match_results)
        expected_column_indices = [2, 4]
        unmatched_column_indices = match.unmatched_column_indices()
        self.assertEqual(unmatched_column_indices.dtype, tf.int32)
        with self.test_session() as sess:
            unmatched_column_indices = sess.run(unmatched_column_indices)
            self.assertAllEqual(unmatched_column_indices, expected_column_indices)

    def testGetCorrectMatchedRowIndices(self):
        if False:
            return 10
        match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
        match = matcher.Match(match_results)
        expected_row_indices = [3, 1, 0, 5]
        matched_row_indices = match.matched_row_indices()
        self.assertEqual(matched_row_indices.dtype, tf.int32)
        with self.test_session() as sess:
            matched_row_inds = sess.run(matched_row_indices)
            self.assertAllEqual(matched_row_inds, expected_row_indices)

    def test_get_correct_ignored_column_indices(self):
        if False:
            print('Hello World!')
        match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
        match = matcher.Match(match_results)
        expected_column_indices = [6]
        ignored_column_indices = match.ignored_column_indices()
        self.assertEqual(ignored_column_indices.dtype, tf.int32)
        with self.test_session() as sess:
            ignored_column_indices = sess.run(ignored_column_indices)
            self.assertAllEqual(ignored_column_indices, expected_column_indices)

    def test_get_correct_matched_column_indicator(self):
        if False:
            while True:
                i = 10
        match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
        match = matcher.Match(match_results)
        expected_column_indicator = [True, True, False, True, False, True, False]
        matched_column_indicator = match.matched_column_indicator()
        self.assertEqual(matched_column_indicator.dtype, tf.bool)
        with self.test_session() as sess:
            matched_column_indicator = sess.run(matched_column_indicator)
            self.assertAllEqual(matched_column_indicator, expected_column_indicator)

    def test_get_correct_unmatched_column_indicator(self):
        if False:
            print('Hello World!')
        match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
        match = matcher.Match(match_results)
        expected_column_indicator = [False, False, True, False, True, False, False]
        unmatched_column_indicator = match.unmatched_column_indicator()
        self.assertEqual(unmatched_column_indicator.dtype, tf.bool)
        with self.test_session() as sess:
            unmatched_column_indicator = sess.run(unmatched_column_indicator)
            self.assertAllEqual(unmatched_column_indicator, expected_column_indicator)

    def test_get_correct_ignored_column_indicator(self):
        if False:
            print('Hello World!')
        match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
        match = matcher.Match(match_results)
        expected_column_indicator = [False, False, False, False, False, False, True]
        ignored_column_indicator = match.ignored_column_indicator()
        self.assertEqual(ignored_column_indicator.dtype, tf.bool)
        with self.test_session() as sess:
            ignored_column_indicator = sess.run(ignored_column_indicator)
            self.assertAllEqual(ignored_column_indicator, expected_column_indicator)

    def test_get_correct_unmatched_ignored_column_indices(self):
        if False:
            print('Hello World!')
        match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
        match = matcher.Match(match_results)
        expected_column_indices = [2, 4, 6]
        unmatched_ignored_column_indices = match.unmatched_or_ignored_column_indices()
        self.assertEqual(unmatched_ignored_column_indices.dtype, tf.int32)
        with self.test_session() as sess:
            unmatched_ignored_column_indices = sess.run(unmatched_ignored_column_indices)
            self.assertAllEqual(unmatched_ignored_column_indices, expected_column_indices)

    def test_all_columns_accounted_for(self):
        if False:
            i = 10
            return i + 15
        num_matches = 10
        match_results = tf.random_uniform([num_matches], minval=-2, maxval=5, dtype=tf.int32)
        match = matcher.Match(match_results)
        matched_column_indices = match.matched_column_indices()
        unmatched_column_indices = match.unmatched_column_indices()
        ignored_column_indices = match.ignored_column_indices()
        with self.test_session() as sess:
            (matched, unmatched, ignored) = sess.run([matched_column_indices, unmatched_column_indices, ignored_column_indices])
            all_indices = np.hstack((matched, unmatched, ignored))
            all_indices_sorted = np.sort(all_indices)
            self.assertAllEqual(all_indices_sorted, np.arange(num_matches, dtype=np.int32))

    def test_scalar_gather_based_on_match(self):
        if False:
            print('Hello World!')
        match_results = tf.constant([3, 1, -1, 0, -1, 5, -2])
        input_tensor = tf.constant([0, 1, 2, 3, 4, 5, 6, 7], dtype=tf.float32)
        expected_gathered_tensor = [3, 1, 100, 0, 100, 5, 200]
        match = matcher.Match(match_results)
        gathered_tensor = match.gather_based_on_match(input_tensor, unmatched_value=100.0, ignored_value=200.0)
        self.assertEqual(gathered_tensor.dtype, tf.float32)
        with self.test_session():
            gathered_tensor_out = gathered_tensor.eval()
        self.assertAllEqual(expected_gathered_tensor, gathered_tensor_out)

    def test_multidimensional_gather_based_on_match(self):
        if False:
            print('Hello World!')
        match_results = tf.constant([1, -1, -2])
        input_tensor = tf.constant([[0, 0.5, 0, 0.5], [0, 0, 0.5, 0.5]], dtype=tf.float32)
        expected_gathered_tensor = [[0, 0, 0.5, 0.5], [0, 0, 0, 0], [0, 0, 0, 0]]
        match = matcher.Match(match_results)
        gathered_tensor = match.gather_based_on_match(input_tensor, unmatched_value=tf.zeros(4), ignored_value=tf.zeros(4))
        self.assertEqual(gathered_tensor.dtype, tf.float32)
        with self.test_session():
            gathered_tensor_out = gathered_tensor.eval()
        self.assertAllEqual(expected_gathered_tensor, gathered_tensor_out)

    def test_multidimensional_gather_based_on_match_with_matmul_gather_op(self):
        if False:
            return 10
        match_results = tf.constant([1, -1, -2])
        input_tensor = tf.constant([[0, 0.5, 0, 0.5], [0, 0, 0.5, 0.5]], dtype=tf.float32)
        expected_gathered_tensor = [[0, 0, 0.5, 0.5], [0, 0, 0, 0], [0, 0, 0, 0]]
        match = matcher.Match(match_results, use_matmul_gather=True)
        gathered_tensor = match.gather_based_on_match(input_tensor, unmatched_value=tf.zeros(4), ignored_value=tf.zeros(4))
        self.assertEqual(gathered_tensor.dtype, tf.float32)
        with self.test_session() as sess:
            self.assertTrue(all([op.name is not 'Gather' for op in sess.graph.get_operations()]))
            gathered_tensor_out = gathered_tensor.eval()
        self.assertAllEqual(expected_gathered_tensor, gathered_tensor_out)
if __name__ == '__main__':
    tf.test.main()