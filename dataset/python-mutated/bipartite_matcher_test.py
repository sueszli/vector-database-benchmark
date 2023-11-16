"""Tests for object_detection.core.bipartite_matcher."""
import tensorflow as tf
from object_detection.matchers import bipartite_matcher

class GreedyBipartiteMatcherTest(tf.test.TestCase):

    def test_get_expected_matches_when_all_rows_are_valid(self):
        if False:
            for i in range(10):
                print('nop')
        similarity_matrix = tf.constant([[0.5, 0.1, 0.8], [0.15, 0.2, 0.3]])
        valid_rows = tf.ones([2], dtype=tf.bool)
        expected_match_results = [-1, 1, 0]
        matcher = bipartite_matcher.GreedyBipartiteMatcher()
        match = matcher.match(similarity_matrix, valid_rows=valid_rows)
        with self.test_session() as sess:
            match_results_out = sess.run(match._match_results)
            self.assertAllEqual(match_results_out, expected_match_results)

    def test_get_expected_matches_with_all_rows_be_default(self):
        if False:
            for i in range(10):
                print('nop')
        similarity_matrix = tf.constant([[0.5, 0.1, 0.8], [0.15, 0.2, 0.3]])
        expected_match_results = [-1, 1, 0]
        matcher = bipartite_matcher.GreedyBipartiteMatcher()
        match = matcher.match(similarity_matrix)
        with self.test_session() as sess:
            match_results_out = sess.run(match._match_results)
            self.assertAllEqual(match_results_out, expected_match_results)

    def test_get_no_matches_with_zero_valid_rows(self):
        if False:
            i = 10
            return i + 15
        similarity_matrix = tf.constant([[0.5, 0.1, 0.8], [0.15, 0.2, 0.3]])
        valid_rows = tf.zeros([2], dtype=tf.bool)
        expected_match_results = [-1, -1, -1]
        matcher = bipartite_matcher.GreedyBipartiteMatcher()
        match = matcher.match(similarity_matrix, valid_rows)
        with self.test_session() as sess:
            match_results_out = sess.run(match._match_results)
            self.assertAllEqual(match_results_out, expected_match_results)

    def test_get_expected_matches_with_only_one_valid_row(self):
        if False:
            return 10
        similarity_matrix = tf.constant([[0.5, 0.1, 0.8], [0.15, 0.2, 0.3]])
        valid_rows = tf.constant([True, False], dtype=tf.bool)
        expected_match_results = [-1, -1, 0]
        matcher = bipartite_matcher.GreedyBipartiteMatcher()
        match = matcher.match(similarity_matrix, valid_rows)
        with self.test_session() as sess:
            match_results_out = sess.run(match._match_results)
            self.assertAllEqual(match_results_out, expected_match_results)

    def test_get_expected_matches_with_only_one_valid_row_at_bottom(self):
        if False:
            print('Hello World!')
        similarity_matrix = tf.constant([[0.15, 0.2, 0.3], [0.5, 0.1, 0.8]])
        valid_rows = tf.constant([False, True], dtype=tf.bool)
        expected_match_results = [-1, -1, 0]
        matcher = bipartite_matcher.GreedyBipartiteMatcher()
        match = matcher.match(similarity_matrix, valid_rows)
        with self.test_session() as sess:
            match_results_out = sess.run(match._match_results)
            self.assertAllEqual(match_results_out, expected_match_results)
if __name__ == '__main__':
    tf.test.main()