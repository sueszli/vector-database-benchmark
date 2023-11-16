import numpy as np
from urh.awre.CommonRange import CommonRange
from urh.awre.engines.Engine import Engine
from urh.cythonext import awre_util

class SequenceNumberEngine(Engine):

    def __init__(self, bitvectors, n_gram_length=8, minimum_score=0.75, already_labeled: list=None):
        if False:
            i = 10
            return i + 15
        '\n\n        :type bitvectors: list of np.ndarray\n        :param bitvectors: bitvectors behind the synchronization\n        '
        self.bitvectors = bitvectors
        self.n_gram_length = n_gram_length
        self.minimum_score = minimum_score
        if already_labeled is None:
            self.already_labeled_cols = set()
        else:
            self.already_labeled_cols = {e // n_gram_length for rng in already_labeled for e in range(*rng)}

    def find(self):
        if False:
            return 10
        n = self.n_gram_length
        if len(self.bitvectors) < 3:
            return []
        diff_matrix = self.create_difference_matrix(self.bitvectors, self.n_gram_length)
        diff_frequencies_by_column = dict()
        for j in range(diff_matrix.shape[1]):
            (unique, counts) = np.unique(diff_matrix[:, j], return_counts=True)
            diff_frequencies_by_column[j] = dict(zip(unique, counts))
        self._debug('Diff_frequencies_by_column', diff_frequencies_by_column)
        scores_by_column = dict()
        for (column, frequencies) in diff_frequencies_by_column.items():
            if column not in self.already_labeled_cols:
                scores_by_column[column] = self.calc_score(frequencies)
            else:
                scores_by_column[column] = 0
        self._debug('Scores by column', scores_by_column)
        result = []
        for candidate_column in sorted(scores_by_column, key=scores_by_column.get, reverse=True):
            score = scores_by_column[candidate_column]
            if score < self.minimum_score:
                continue
            most_common_diff = self.get_most_frequent(diff_frequencies_by_column[candidate_column])
            message_indices = np.flatnonzero((diff_matrix[:, candidate_column] == most_common_diff) | (diff_matrix[:, candidate_column] == 0))
            message_indices = set(message_indices) | set(message_indices + 1)
            values = set()
            for i in message_indices:
                values.add(self.bitvectors[i][candidate_column * n:(candidate_column + 1) * n].tobytes())
            matching_ranges = [r for r in result if r.message_indices == message_indices]
            try:
                matching_range = next((r for r in matching_ranges if r.start == (candidate_column - 1) * n and (r.byte_order_is_unknown or r.byte_order == 'big')))
                matching_range.length += n
                matching_range.byte_order = 'big'
                matching_range.values.extend(list(values))
                continue
            except StopIteration:
                pass
            try:
                matching_range = next((r for r in matching_ranges if r.start == (candidate_column + 1) * n and (r.byte_order_is_unknown or r.byte_order == 'little')))
                matching_range.start -= n
                matching_range.length += n
                matching_range.byte_order = 'little'
                matching_range.values.extend(list(values))
                continue
            except StopIteration:
                pass
            new_range = CommonRange(start=candidate_column * n, length=n, score=score, field_type='sequence number', message_indices=message_indices, byte_order=None)
            new_range.values.extend(list(values))
            result.append(new_range)
        return [rng for rng in result if len(set(rng.values)) > 2]

    @staticmethod
    def get_most_frequent(diff_frequencies: dict):
        if False:
            for i in range(10):
                print('nop')
        return max(filter(lambda x: x not in (0, -1), diff_frequencies), key=diff_frequencies.get)

    @staticmethod
    def calc_score(diff_frequencies: dict) -> float:
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculate the score based on the distribution of differences\n          1. high if one constant (!= zero) dominates\n          2. Other constants (!= zero) should lower the score, zero means sequence number stays same for some messages\n\n        :param diff_frequencies: Frequencies of decimal differences between columns of subsequent messages\n                                 e.g. {0: 3, 1: 1020} means 0 appeared 3 times and 1 appeared 1020 times\n        :return: a score between 0 and 1\n        '
        total = sum(diff_frequencies.values())
        num_zeros = sum((v for (k, v) in diff_frequencies.items() if k == 0))
        if num_zeros == total:
            return 0
        try:
            most_frequent = SequenceNumberEngine.get_most_frequent(diff_frequencies)
        except ValueError:
            return 0
        return diff_frequencies[most_frequent] / (total - num_zeros)

    @staticmethod
    def create_difference_matrix(bitvectors, n_gram_length: int):
        if False:
            print('Hello World!')
        '\n        Create the difference matrix e.g.\n        10 20 0\n        1  2  3\n        4  5  6\n\n        means first eight bits of messages 1 and 2 (row 1) differ by 10 if they are considered as decimal number\n\n        :type bitvectors: list of np.ndarray\n        :type n_gram_length: int\n        :rtype: np.ndarray\n        '
        return awre_util.create_seq_number_difference_matrix(bitvectors, n_gram_length)