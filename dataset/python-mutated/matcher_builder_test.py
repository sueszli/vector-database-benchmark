"""Tests for matcher_builder."""
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import matcher_builder
from object_detection.matchers import argmax_matcher
from object_detection.matchers import bipartite_matcher
from object_detection.protos import matcher_pb2

class MatcherBuilderTest(tf.test.TestCase):

    def test_build_arg_max_matcher_with_defaults(self):
        if False:
            while True:
                i = 10
        matcher_text_proto = '\n      argmax_matcher {\n      }\n    '
        matcher_proto = matcher_pb2.Matcher()
        text_format.Merge(matcher_text_proto, matcher_proto)
        matcher_object = matcher_builder.build(matcher_proto)
        self.assertTrue(isinstance(matcher_object, argmax_matcher.ArgMaxMatcher))
        self.assertAlmostEqual(matcher_object._matched_threshold, 0.5)
        self.assertAlmostEqual(matcher_object._unmatched_threshold, 0.5)
        self.assertTrue(matcher_object._negatives_lower_than_unmatched)
        self.assertFalse(matcher_object._force_match_for_each_row)

    def test_build_arg_max_matcher_without_thresholds(self):
        if False:
            print('Hello World!')
        matcher_text_proto = '\n      argmax_matcher {\n        ignore_thresholds: true\n      }\n    '
        matcher_proto = matcher_pb2.Matcher()
        text_format.Merge(matcher_text_proto, matcher_proto)
        matcher_object = matcher_builder.build(matcher_proto)
        self.assertTrue(isinstance(matcher_object, argmax_matcher.ArgMaxMatcher))
        self.assertEqual(matcher_object._matched_threshold, None)
        self.assertEqual(matcher_object._unmatched_threshold, None)
        self.assertTrue(matcher_object._negatives_lower_than_unmatched)
        self.assertFalse(matcher_object._force_match_for_each_row)

    def test_build_arg_max_matcher_with_non_default_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        matcher_text_proto = '\n      argmax_matcher {\n        matched_threshold: 0.7\n        unmatched_threshold: 0.3\n        negatives_lower_than_unmatched: false\n        force_match_for_each_row: true\n        use_matmul_gather: true\n      }\n    '
        matcher_proto = matcher_pb2.Matcher()
        text_format.Merge(matcher_text_proto, matcher_proto)
        matcher_object = matcher_builder.build(matcher_proto)
        self.assertTrue(isinstance(matcher_object, argmax_matcher.ArgMaxMatcher))
        self.assertAlmostEqual(matcher_object._matched_threshold, 0.7)
        self.assertAlmostEqual(matcher_object._unmatched_threshold, 0.3)
        self.assertFalse(matcher_object._negatives_lower_than_unmatched)
        self.assertTrue(matcher_object._force_match_for_each_row)
        self.assertTrue(matcher_object._use_matmul_gather)

    def test_build_bipartite_matcher(self):
        if False:
            return 10
        matcher_text_proto = '\n      bipartite_matcher {\n      }\n    '
        matcher_proto = matcher_pb2.Matcher()
        text_format.Merge(matcher_text_proto, matcher_proto)
        matcher_object = matcher_builder.build(matcher_proto)
        self.assertTrue(isinstance(matcher_object, bipartite_matcher.GreedyBipartiteMatcher))

    def test_raise_error_on_empty_matcher(self):
        if False:
            return 10
        matcher_text_proto = '\n    '
        matcher_proto = matcher_pb2.Matcher()
        text_format.Merge(matcher_text_proto, matcher_proto)
        with self.assertRaises(ValueError):
            matcher_builder.build(matcher_proto)
if __name__ == '__main__':
    tf.test.main()