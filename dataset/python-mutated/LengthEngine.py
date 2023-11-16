import math
from collections import defaultdict
import numpy as np
from urh.awre.CommonRange import CommonRange
from urh.awre.engines.Engine import Engine
from urh.cythonext import util

class LengthEngine(Engine):

    def __init__(self, bitvectors, already_labeled=None):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        :type bitvectors: list of np.ndarray\n        :param bitvectors: bitvectors behind the synchronization\n        '
        self.bitvectors = bitvectors
        self.already_labeled = [] if already_labeled is None else already_labeled

    def find(self, n_gram_length=8, minimum_score=0.1):
        if False:
            print('Hello World!')
        bitvectors_by_n_gram_length = defaultdict(list)
        for (i, bitvector) in enumerate(self.bitvectors):
            bin_num = int(math.ceil(len(bitvector) / n_gram_length))
            bitvectors_by_n_gram_length[bin_num].append(i)
        common_ranges_by_length = self.find_common_ranges_by_cluster(self.bitvectors, bitvectors_by_n_gram_length, alpha=0.7)
        for (length, ranges) in common_ranges_by_length.items():
            common_ranges_by_length[length] = self.ignore_already_labeled(ranges, self.already_labeled)
        self.filter_common_ranges(common_ranges_by_length)
        self._debug('Common Ranges:', common_ranges_by_length)
        scored_ranges = self.score_ranges(common_ranges_by_length, n_gram_length)
        self._debug('Scored Ranges', scored_ranges)
        high_scores_by_length = self.choose_high_scored_ranges(scored_ranges, bitvectors_by_n_gram_length, minimum_score)
        self._debug('Highscored Ranges', high_scores_by_length)
        return high_scores_by_length.values()

    @staticmethod
    def filter_common_ranges(common_ranges_by_length: dict):
        if False:
            print('Hello World!')
        '\n        Ranges must be common along length clusters\n        but their values must differ, so now we rule out all ranges that are\n          1. common across clusters AND\n          2. have same value\n\n        :return:\n        '
        ranges = [r for rng in common_ranges_by_length.values() for r in rng]
        for rng in ranges:
            count = len([r for r in ranges if rng.start == r.start and rng.length == r.length and (rng.value.tobytes() == r.value.tobytes())])
            if count < 2:
                continue
            for length in common_ranges_by_length:
                try:
                    common_ranges_by_length[length].remove(rng)
                except ValueError:
                    pass

    @staticmethod
    def score_ranges(common_ranges_by_length: dict, n_gram_length: int):
        if False:
            print('Hello World!')
        '\n        Calculate score for the common ranges\n\n        :param common_ranges_by_length:\n        :param n_gram_length:\n        :return:\n        '
        if n_gram_length == 8:
            window_lengths = [8, 16, 32, 64]
        else:
            window_lengths = [n_gram_length * i for i in range(1, 5)]
        scored_ranges = dict()
        for length in common_ranges_by_length:
            scored_ranges[length] = dict()
            for window_length in window_lengths:
                scored_ranges[length][window_length] = []
        byteorders = ['big', 'little'] if n_gram_length == 8 else ['big']
        for window_length in window_lengths:
            for (length, common_ranges) in common_ranges_by_length.items():
                for common_range in filter(lambda cr: cr.length >= window_length, common_ranges):
                    bits = common_range.value
                    rng_byte_order = 'big'
                    max_score = max_start = -1
                    for start in range(0, len(bits) + 1 - window_length, n_gram_length):
                        for byteorder in byteorders:
                            score = LengthEngine.score_bits(bits[start:start + window_length], length, position=start, byteorder=byteorder)
                            if score > max_score:
                                max_score = score
                                max_start = start
                                rng_byte_order = byteorder
                    rng = CommonRange(common_range.start + max_start, window_length, common_range.value[max_start:max_start + window_length], score=max_score, field_type='length', message_indices=common_range.message_indices, range_type=common_range.range_type, byte_order=rng_byte_order)
                    scored_ranges[length][window_length].append(rng)
        return scored_ranges

    def choose_high_scored_ranges(self, scored_ranges: dict, bitvectors_by_n_gram_length: dict, minimum_score: float):
        if False:
            return 10
        possible_window_lengths = defaultdict(int)
        for (length, ranges_by_window_length) in scored_ranges.items():
            for (window_length, ranges) in ranges_by_window_length.items():
                try:
                    ranges_by_window_length[window_length] = max(filter(lambda x: x.score >= minimum_score, ranges), key=lambda x: x.score)
                    possible_window_lengths[window_length] += 1
                except ValueError:
                    ranges_by_window_length[window_length] = None
        try:
            chosen_window_length = max(possible_window_lengths, key=lambda x: (possible_window_lengths[x], x))
        except ValueError:
            return dict()
        high_scores_by_length = dict()
        for (length, ranges_by_window_length) in scored_ranges.items():
            try:
                if ranges_by_window_length[chosen_window_length]:
                    high_scores_by_length[length] = ranges_by_window_length[chosen_window_length]
            except KeyError:
                continue
        for (length, msg_indices) in bitvectors_by_n_gram_length.items():
            if len(msg_indices) != 1:
                continue
            msg_index = msg_indices[0]
            bitvector = self.bitvectors[msg_index]
            (max_score, best_match) = (0, None)
            for rng in high_scores_by_length.values():
                bits = bitvector[rng.start:rng.end + 1]
                if len(bits) > 0:
                    score = self.score_bits(bits, length, rng.start)
                    if score > max_score:
                        (best_match, max_score) = (rng, score)
            if best_match is not None:
                high_scores_by_length[length] = CommonRange(best_match.start, best_match.length, value=bitvector[best_match.start:best_match.end + 1], score=max_score, field_type='length', message_indices={msg_index}, range_type='bit')
        return high_scores_by_length

    @staticmethod
    def score_bits(bits: np.ndarray, target_length: int, position: int, byteorder='big'):
        if False:
            print('Hello World!')
        value = util.bit_array_to_number(bits, len(bits))
        if byteorder == 'little':
            if len(bits) > 8 and len(bits) % 8 == 0:
                n = len(bits) // 8
                value = int.from_bytes(value.to_bytes(n, byteorder='big'), byteorder='little', signed=False)
        f = 1 / (1 + 0.25 * position)
        return f * LengthEngine.gauss(value, target_length)

    @staticmethod
    def gauss(x, mu, sigma=2):
        if False:
            for i in range(10):
                print('nop')
        return np.exp(-0.5 * np.power((x - mu) / sigma, 2))