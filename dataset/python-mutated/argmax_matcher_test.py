"""Tests for object_detection.matchers.argmax_matcher."""
import numpy as np
import tensorflow as tf
from object_detection.matchers import argmax_matcher
from object_detection.utils import test_case

class ArgMaxMatcherTest(test_case.TestCase):

    def test_return_correct_matches_with_default_thresholds(self):
        if False:
            while True:
                i = 10

        def graph_fn(similarity_matrix):
            if False:
                print('Hello World!')
            matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=None)
            match = matcher.match(similarity_matrix)
            matched_cols = match.matched_column_indicator()
            unmatched_cols = match.unmatched_column_indicator()
            match_results = match.match_results
            return (matched_cols, unmatched_cols, match_results)
        similarity = np.array([[1.0, 1, 1, 3, 1], [2, -1, 2, 0, 4], [3, 0, -1, 0, 0]], dtype=np.float32)
        expected_matched_rows = np.array([2, 0, 1, 0, 1])
        (res_matched_cols, res_unmatched_cols, res_match_results) = self.execute(graph_fn, [similarity])
        self.assertAllEqual(res_match_results[res_matched_cols], expected_matched_rows)
        self.assertAllEqual(np.nonzero(res_matched_cols)[0], [0, 1, 2, 3, 4])
        self.assertFalse(np.all(res_unmatched_cols))

    def test_return_correct_matches_with_empty_rows(self):
        if False:
            while True:
                i = 10

        def graph_fn(similarity_matrix):
            if False:
                while True:
                    i = 10
            matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=None)
            match = matcher.match(similarity_matrix)
            return match.unmatched_column_indicator()
        similarity = 0.2 * np.ones([0, 5], dtype=np.float32)
        res_unmatched_cols = self.execute(graph_fn, [similarity])
        self.assertAllEqual(np.nonzero(res_unmatched_cols)[0], np.arange(5))

    def test_return_correct_matches_with_matched_threshold(self):
        if False:
            print('Hello World!')

        def graph_fn(similarity):
            if False:
                return 10
            matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=3.0)
            match = matcher.match(similarity)
            matched_cols = match.matched_column_indicator()
            unmatched_cols = match.unmatched_column_indicator()
            match_results = match.match_results
            return (matched_cols, unmatched_cols, match_results)
        similarity = np.array([[1, 1, 1, 3, 1], [2, -1, 2, 0, 4], [3, 0, -1, 0, 0]], dtype=np.float32)
        expected_matched_cols = np.array([0, 3, 4])
        expected_matched_rows = np.array([2, 0, 1])
        expected_unmatched_cols = np.array([1, 2])
        (res_matched_cols, res_unmatched_cols, match_results) = self.execute(graph_fn, [similarity])
        self.assertAllEqual(match_results[res_matched_cols], expected_matched_rows)
        self.assertAllEqual(np.nonzero(res_matched_cols)[0], expected_matched_cols)
        self.assertAllEqual(np.nonzero(res_unmatched_cols)[0], expected_unmatched_cols)

    def test_return_correct_matches_with_matched_and_unmatched_threshold(self):
        if False:
            return 10

        def graph_fn(similarity):
            if False:
                return 10
            matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=3.0, unmatched_threshold=2.0)
            match = matcher.match(similarity)
            matched_cols = match.matched_column_indicator()
            unmatched_cols = match.unmatched_column_indicator()
            match_results = match.match_results
            return (matched_cols, unmatched_cols, match_results)
        similarity = np.array([[1, 1, 1, 3, 1], [2, -1, 2, 0, 4], [3, 0, -1, 0, 0]], dtype=np.float32)
        expected_matched_cols = np.array([0, 3, 4])
        expected_matched_rows = np.array([2, 0, 1])
        expected_unmatched_cols = np.array([1])
        (res_matched_cols, res_unmatched_cols, match_results) = self.execute(graph_fn, [similarity])
        self.assertAllEqual(match_results[res_matched_cols], expected_matched_rows)
        self.assertAllEqual(np.nonzero(res_matched_cols)[0], expected_matched_cols)
        self.assertAllEqual(np.nonzero(res_unmatched_cols)[0], expected_unmatched_cols)

    def test_return_correct_matches_negatives_lower_than_unmatched_false(self):
        if False:
            print('Hello World!')

        def graph_fn(similarity):
            if False:
                print('Hello World!')
            matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=3.0, unmatched_threshold=2.0, negatives_lower_than_unmatched=False)
            match = matcher.match(similarity)
            matched_cols = match.matched_column_indicator()
            unmatched_cols = match.unmatched_column_indicator()
            match_results = match.match_results
            return (matched_cols, unmatched_cols, match_results)
        similarity = np.array([[1, 1, 1, 3, 1], [2, -1, 2, 0, 4], [3, 0, -1, 0, 0]], dtype=np.float32)
        expected_matched_cols = np.array([0, 3, 4])
        expected_matched_rows = np.array([2, 0, 1])
        expected_unmatched_cols = np.array([2])
        (res_matched_cols, res_unmatched_cols, match_results) = self.execute(graph_fn, [similarity])
        self.assertAllEqual(match_results[res_matched_cols], expected_matched_rows)
        self.assertAllEqual(np.nonzero(res_matched_cols)[0], expected_matched_cols)
        self.assertAllEqual(np.nonzero(res_unmatched_cols)[0], expected_unmatched_cols)

    def test_return_correct_matches_unmatched_row_not_using_force_match(self):
        if False:
            print('Hello World!')

        def graph_fn(similarity):
            if False:
                while True:
                    i = 10
            matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=3.0, unmatched_threshold=2.0)
            match = matcher.match(similarity)
            matched_cols = match.matched_column_indicator()
            unmatched_cols = match.unmatched_column_indicator()
            match_results = match.match_results
            return (matched_cols, unmatched_cols, match_results)
        similarity = np.array([[1, 1, 1, 3, 1], [-1, 0, -2, -2, -1], [3, 0, -1, 2, 0]], dtype=np.float32)
        expected_matched_cols = np.array([0, 3])
        expected_matched_rows = np.array([2, 0])
        expected_unmatched_cols = np.array([1, 2, 4])
        (res_matched_cols, res_unmatched_cols, match_results) = self.execute(graph_fn, [similarity])
        self.assertAllEqual(match_results[res_matched_cols], expected_matched_rows)
        self.assertAllEqual(np.nonzero(res_matched_cols)[0], expected_matched_cols)
        self.assertAllEqual(np.nonzero(res_unmatched_cols)[0], expected_unmatched_cols)

    def test_return_correct_matches_unmatched_row_while_using_force_match(self):
        if False:
            return 10

        def graph_fn(similarity):
            if False:
                for i in range(10):
                    print('nop')
            matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=3.0, unmatched_threshold=2.0, force_match_for_each_row=True)
            match = matcher.match(similarity)
            matched_cols = match.matched_column_indicator()
            unmatched_cols = match.unmatched_column_indicator()
            match_results = match.match_results
            return (matched_cols, unmatched_cols, match_results)
        similarity = np.array([[1, 1, 1, 3, 1], [-1, 0, -2, -2, -1], [3, 0, -1, 2, 0]], dtype=np.float32)
        expected_matched_cols = np.array([0, 1, 3])
        expected_matched_rows = np.array([2, 1, 0])
        expected_unmatched_cols = np.array([2, 4])
        (res_matched_cols, res_unmatched_cols, match_results) = self.execute(graph_fn, [similarity])
        self.assertAllEqual(match_results[res_matched_cols], expected_matched_rows)
        self.assertAllEqual(np.nonzero(res_matched_cols)[0], expected_matched_cols)
        self.assertAllEqual(np.nonzero(res_unmatched_cols)[0], expected_unmatched_cols)

    def test_return_correct_matches_using_force_match_padded_groundtruth(self):
        if False:
            print('Hello World!')

        def graph_fn(similarity, valid_rows):
            if False:
                while True:
                    i = 10
            matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=3.0, unmatched_threshold=2.0, force_match_for_each_row=True)
            match = matcher.match(similarity, valid_rows)
            matched_cols = match.matched_column_indicator()
            unmatched_cols = match.unmatched_column_indicator()
            match_results = match.match_results
            return (matched_cols, unmatched_cols, match_results)
        similarity = np.array([[1, 1, 1, 3, 1], [-1, 0, -2, -2, -1], [0, 0, 0, 0, 0], [3, 0, -1, 2, 0], [0, 0, 0, 0, 0]], dtype=np.float32)
        valid_rows = np.array([True, True, False, True, False])
        expected_matched_cols = np.array([0, 1, 3])
        expected_matched_rows = np.array([3, 1, 0])
        expected_unmatched_cols = np.array([2, 4])
        (res_matched_cols, res_unmatched_cols, match_results) = self.execute(graph_fn, [similarity, valid_rows])
        self.assertAllEqual(match_results[res_matched_cols], expected_matched_rows)
        self.assertAllEqual(np.nonzero(res_matched_cols)[0], expected_matched_cols)
        self.assertAllEqual(np.nonzero(res_unmatched_cols)[0], expected_unmatched_cols)

    def test_valid_arguments_corner_case(self):
        if False:
            print('Hello World!')
        argmax_matcher.ArgMaxMatcher(matched_threshold=1, unmatched_threshold=1)

    def test_invalid_arguments_corner_case_negatives_lower_than_thres_false(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            argmax_matcher.ArgMaxMatcher(matched_threshold=1, unmatched_threshold=1, negatives_lower_than_unmatched=False)

    def test_invalid_arguments_no_matched_threshold(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            argmax_matcher.ArgMaxMatcher(matched_threshold=None, unmatched_threshold=4)

    def test_invalid_arguments_unmatched_thres_larger_than_matched_thres(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            argmax_matcher.ArgMaxMatcher(matched_threshold=1, unmatched_threshold=2)
if __name__ == '__main__':
    tf.test.main()