"""A function to build an object detection matcher from configuration."""
from object_detection.matchers import argmax_matcher
from object_detection.matchers import bipartite_matcher
from object_detection.protos import matcher_pb2

def build(matcher_config):
    if False:
        i = 10
        return i + 15
    'Builds a matcher object based on the matcher config.\n\n  Args:\n    matcher_config: A matcher.proto object containing the config for the desired\n      Matcher.\n\n  Returns:\n    Matcher based on the config.\n\n  Raises:\n    ValueError: On empty matcher proto.\n  '
    if not isinstance(matcher_config, matcher_pb2.Matcher):
        raise ValueError('matcher_config not of type matcher_pb2.Matcher.')
    if matcher_config.WhichOneof('matcher_oneof') == 'argmax_matcher':
        matcher = matcher_config.argmax_matcher
        matched_threshold = unmatched_threshold = None
        if not matcher.ignore_thresholds:
            matched_threshold = matcher.matched_threshold
            unmatched_threshold = matcher.unmatched_threshold
        return argmax_matcher.ArgMaxMatcher(matched_threshold=matched_threshold, unmatched_threshold=unmatched_threshold, negatives_lower_than_unmatched=matcher.negatives_lower_than_unmatched, force_match_for_each_row=matcher.force_match_for_each_row, use_matmul_gather=matcher.use_matmul_gather)
    if matcher_config.WhichOneof('matcher_oneof') == 'bipartite_matcher':
        matcher = matcher_config.bipartite_matcher
        return bipartite_matcher.GreedyBipartiteMatcher(matcher.use_matmul_gather)
    raise ValueError('Empty matcher.')