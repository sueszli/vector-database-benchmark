"""Matcher interface and Match class.

This module defines the Matcher interface and the Match object. The job of the
matcher is to match row and column indices based on the similarity matrix and
other optional parameters. Each column is matched to at most one row. There
are three possibilities for the matching:

1) match: A column matches a row.
2) no_match: A column does not match any row.
3) ignore: A column that is neither 'match' nor no_match.

The ignore case is regularly encountered in object detection: when an anchor has
a relatively small overlap with a ground-truth box, one neither wants to
consider this box a positive example (match) nor a negative example (no match).

The Match class is used to store the match results and it provides simple apis
to query the results.
"""
from abc import ABCMeta
from abc import abstractmethod
import tensorflow.compat.v2 as tf

class Match(object):
    """Class to store results from the matcher.

  This class is used to store the results from the matcher. It provides
  convenient methods to query the matching results.
  """

    def __init__(self, match_results):
        if False:
            i = 10
            return i + 15
        'Constructs a Match object.\n\n    Args:\n      match_results: Integer tensor of shape [N] with (1) match_results[i]>=0,\n        meaning that column i is matched with row match_results[i].\n        (2) match_results[i]=-1, meaning that column i is not matched.\n        (3) match_results[i]=-2, meaning that column i is ignored.\n\n    Raises:\n      ValueError: if match_results does not have rank 1 or is not an\n        integer int32 scalar tensor\n    '
        if match_results.shape.ndims != 1:
            raise ValueError('match_results should have rank 1')
        if match_results.dtype != tf.int32:
            raise ValueError('match_results should be an int32 or int64 scalar tensor')
        self._match_results = match_results

    @property
    def match_results(self):
        if False:
            while True:
                i = 10
        'The accessor for match results.\n\n    Returns:\n      the tensor which encodes the match results.\n    '
        return self._match_results

    def matched_column_indices(self):
        if False:
            print('Hello World!')
        'Returns column indices that match to some row.\n\n    The indices returned by this op are always sorted in increasing order.\n\n    Returns:\n      column_indices: int32 tensor of shape [K] with column indices.\n    '
        return self._reshape_and_cast(tf.where(tf.greater(self._match_results, -1)))

    def matched_column_indicator(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns column indices that are matched.\n\n    Returns:\n      column_indices: int32 tensor of shape [K] with column indices.\n    '
        return tf.greater_equal(self._match_results, 0)

    def num_matched_columns(self):
        if False:
            i = 10
            return i + 15
        'Returns number (int32 scalar tensor) of matched columns.'
        return tf.size(input=self.matched_column_indices())

    def unmatched_column_indices(self):
        if False:
            return 10
        'Returns column indices that do not match any row.\n\n    The indices returned by this op are always sorted in increasing order.\n\n    Returns:\n      column_indices: int32 tensor of shape [K] with column indices.\n    '
        return self._reshape_and_cast(tf.where(tf.equal(self._match_results, -1)))

    def unmatched_column_indicator(self):
        if False:
            i = 10
            return i + 15
        'Returns column indices that are unmatched.\n\n    Returns:\n      column_indices: int32 tensor of shape [K] with column indices.\n    '
        return tf.equal(self._match_results, -1)

    def num_unmatched_columns(self):
        if False:
            return 10
        'Returns number (int32 scalar tensor) of unmatched columns.'
        return tf.size(input=self.unmatched_column_indices())

    def ignored_column_indices(self):
        if False:
            while True:
                i = 10
        'Returns column indices that are ignored (neither Matched nor Unmatched).\n\n    The indices returned by this op are always sorted in increasing order.\n\n    Returns:\n      column_indices: int32 tensor of shape [K] with column indices.\n    '
        return self._reshape_and_cast(tf.where(self.ignored_column_indicator()))

    def ignored_column_indicator(self):
        if False:
            return 10
        'Returns boolean column indicator where True means the colum is ignored.\n\n    Returns:\n      column_indicator: boolean vector which is True for all ignored column\n      indices.\n    '
        return tf.equal(self._match_results, -2)

    def num_ignored_columns(self):
        if False:
            return 10
        'Returns number (int32 scalar tensor) of matched columns.'
        return tf.size(input=self.ignored_column_indices())

    def unmatched_or_ignored_column_indices(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns column indices that are unmatched or ignored.\n\n    The indices returned by this op are always sorted in increasing order.\n\n    Returns:\n      column_indices: int32 tensor of shape [K] with column indices.\n    '
        return self._reshape_and_cast(tf.where(tf.greater(0, self._match_results)))

    def matched_row_indices(self):
        if False:
            while True:
                i = 10
        'Returns row indices that match some column.\n\n    The indices returned by this op are ordered so as to be in correspondence\n    with the output of matched_column_indicator().  For example if\n    self.matched_column_indicator() is [0,2], and self.matched_row_indices() is\n    [7, 3], then we know that column 0 was matched to row 7 and column 2 was\n    matched to row 3.\n\n    Returns:\n      row_indices: int32 tensor of shape [K] with row indices.\n    '
        return self._reshape_and_cast(tf.gather(self._match_results, self.matched_column_indices()))

    def _reshape_and_cast(self, t):
        if False:
            for i in range(10):
                print('nop')
        return tf.cast(tf.reshape(t, [-1]), tf.int32)

    def gather_based_on_match(self, input_tensor, unmatched_value, ignored_value):
        if False:
            for i in range(10):
                print('nop')
        'Gathers elements from `input_tensor` based on match results.\n\n    For columns that are matched to a row, gathered_tensor[col] is set to\n    input_tensor[match_results[col]]. For columns that are unmatched,\n    gathered_tensor[col] is set to unmatched_value. Finally, for columns that\n    are ignored gathered_tensor[col] is set to ignored_value.\n\n    Note that the input_tensor.shape[1:] must match with unmatched_value.shape\n    and ignored_value.shape\n\n    Args:\n      input_tensor: Tensor to gather values from.\n      unmatched_value: Constant tensor value for unmatched columns.\n      ignored_value: Constant tensor value for ignored columns.\n\n    Returns:\n      gathered_tensor: A tensor containing values gathered from input_tensor.\n        The shape of the gathered tensor is [match_results.shape[0]] +\n        input_tensor.shape[1:].\n    '
        input_tensor = tf.concat([tf.stack([ignored_value, unmatched_value]), input_tensor], axis=0)
        gather_indices = tf.maximum(self.match_results + 2, 0)
        gathered_tensor = tf.gather(input_tensor, gather_indices)
        return gathered_tensor

class Matcher(object):
    """Abstract base class for matcher.
  """
    __metaclass__ = ABCMeta

    def match(self, similarity_matrix, scope=None, **params):
        if False:
            for i in range(10):
                print('nop')
        "Computes matches among row and column indices and returns the result.\n\n    Computes matches among the row and column indices based on the similarity\n    matrix and optional arguments.\n\n    Args:\n      similarity_matrix: Float tensor of shape [N, M] with pairwise similarity\n        where higher value means more similar.\n      scope: Op scope name. Defaults to 'Match' if None.\n      **params: Additional keyword arguments for specific implementations of\n        the Matcher.\n\n    Returns:\n      A Match object with the results of matching.\n    "
        if not scope:
            scope = 'Match'
        with tf.name_scope(scope) as scope:
            return Match(self._match(similarity_matrix, **params))

    @abstractmethod
    def _match(self, similarity_matrix, **params):
        if False:
            while True:
                i = 10
        'Method to be overridden by implementations.\n\n    Args:\n      similarity_matrix: Float tensor of shape [N, M] with pairwise similarity\n        where higher value means more similar.\n      **params: Additional keyword arguments for specific implementations of\n        the Matcher.\n\n    Returns:\n      match_results: Integer tensor of shape [M]: match_results[i]>=0 means\n        that column i is matched to row match_results[i], match_results[i]=-1\n        means that the column is not matched. match_results[i]=-2 means that\n        the column is ignored (usually this happens when there is a very weak\n        match which one neither wants as positive nor negative example).\n    '
        pass