from __future__ import annotations
from collections import defaultdict
from itertools import zip_longest
from typing import Any, Sequence, TypeVar
from .base import Base as _Base, BaseSimilarity as _BaseSimilarity
from .types import SimFunc, TestFunc
try:
    import numpy
except ImportError:
    numpy = None
__all__ = ['Hamming', 'MLIPNS', 'Levenshtein', 'DamerauLevenshtein', 'Jaro', 'JaroWinkler', 'StrCmp95', 'NeedlemanWunsch', 'Gotoh', 'SmithWaterman', 'hamming', 'mlipns', 'levenshtein', 'damerau_levenshtein', 'jaro', 'jaro_winkler', 'strcmp95', 'needleman_wunsch', 'gotoh', 'smith_waterman']
T = TypeVar('T')

class Hamming(_Base):
    """
    Compute the Hamming distance between the two or more sequences.
    The Hamming distance is the number of differing items in ordered sequences.

    https://en.wikipedia.org/wiki/Hamming_distance
    """

    def __init__(self, qval: int=1, test_func: TestFunc | None=None, truncate: bool=False, external: bool=True) -> None:
        if False:
            while True:
                i = 10
        self.qval = qval
        self.test_func = test_func or self._ident
        self.truncate = truncate
        self.external = external

    def __call__(self, *sequences: Sequence[object]) -> int:
        if False:
            i = 10
            return i + 15
        sequences = self._get_sequences(*sequences)
        result = self.quick_answer(*sequences)
        if result is not None:
            assert isinstance(result, int)
            return result
        _zip = zip if self.truncate else zip_longest
        return sum((not self.test_func(*es) for es in _zip(*sequences)))

class Levenshtein(_Base):
    """
    Compute the absolute Levenshtein distance between the two sequences.
    The Levenshtein distance is the minimum number of edit operations necessary
    for transforming one sequence into the other. The edit operations allowed are:

        * deletion:     ABC -> BC, AC, AB
        * insertion:    ABC -> ABCD, EABC, AEBC..
        * substitution: ABC -> ABE, ADC, FBC..

    https://en.wikipedia.org/wiki/Levenshtein_distance
    TODO: https://gist.github.com/kylebgorman/1081951/9b38b7743a3cb5167ab2c6608ac8eea7fc629dca
    """

    def __init__(self, qval: int=1, test_func: TestFunc | None=None, external: bool=True) -> None:
        if False:
            return 10
        self.qval = qval
        self.test_func = test_func or self._ident
        self.external = external

    def _recursive(self, s1: Sequence[T], s2: Sequence[T]) -> int:
        if False:
            return 10
        if not s1 or not s2:
            return len(s1) + len(s2)
        if self.test_func(s1[-1], s2[-1]):
            return self(s1[:-1], s2[:-1])
        d = min(self(s1[:-1], s2), self(s1, s2[:-1]))
        s = self(s1[:-1], s2[:-1])
        return min(d, s) + 1

    def _cycled(self, s1: Sequence[T], s2: Sequence[T]) -> int:
        if False:
            while True:
                i = 10
        '\n        source:\n        https://github.com/jamesturk/jellyfish/blob/master/jellyfish/_jellyfish.py#L18\n        '
        rows = len(s1) + 1
        cols = len(s2) + 1
        prev = None
        cur: Any
        if numpy:
            cur = numpy.arange(cols)
        else:
            cur = range(cols)
        for r in range(1, rows):
            (prev, cur) = (cur, [r] + [0] * (cols - 1))
            for c in range(1, cols):
                deletion = prev[c] + 1
                insertion = cur[c - 1] + 1
                dist = self.test_func(s1[r - 1], s2[c - 1])
                edit = prev[c - 1] + (not dist)
                cur[c] = min(edit, deletion, insertion)
        return cur[-1]

    def __call__(self, s1: Sequence[T], s2: Sequence[T]) -> int:
        if False:
            while True:
                i = 10
        (s1, s2) = self._get_sequences(s1, s2)
        result = self.quick_answer(s1, s2)
        if result is not None:
            assert isinstance(result, int)
            return result
        return self._cycled(s1, s2)

class DamerauLevenshtein(_Base):
    """
    Compute the absolute Damerau-Levenshtein distance between the two sequences.
    The Damerau-Levenshtein distance is the minimum number of edit operations necessary
    for transforming one sequence into the other. The edit operations allowed are:

        * deletion:      ABC -> BC, AC, AB
        * insertion:     ABC -> ABCD, EABC, AEBC..
        * substitution:  ABC -> ABE, ADC, FBC..
        * transposition: ABC -> ACB, BAC

    If `restricted=False`, it will calculate unrestricted distance,
    where the same character can be touched more than once.
    So the distance between BA and ACB is 2: BA -> AB -> ACB.

    https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
    """

    def __init__(self, qval: int=1, test_func: TestFunc | None=None, external: bool=True, restricted: bool=True) -> None:
        if False:
            print('Hello World!')
        self.qval = qval
        self.test_func = test_func or self._ident
        self.external = external
        self.restricted = restricted

    def _numpy(self, s1: Sequence[T], s2: Sequence[T]) -> int:
        if False:
            for i in range(10):
                print('nop')
        d = numpy.zeros([len(s1) + 1, len(s2) + 1], dtype=int)
        for i in range(-1, len(s1) + 1):
            d[i][-1] = i + 1
        for j in range(-1, len(s2) + 1):
            d[-1][j] = j + 1
        for (i, cs1) in enumerate(s1):
            for (j, cs2) in enumerate(s2):
                cost = int(not self.test_func(cs1, cs2))
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
                if not i or not j:
                    continue
                if not self.test_func(cs1, s2[j - 1]):
                    continue
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
        return d[len(s1) - 1][len(s2) - 1]

    def _pure_python_unrestricted(self, s1: Sequence[T], s2: Sequence[T]) -> int:
        if False:
            print('Hello World!')
        'https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance\n        '
        d: dict[tuple[int, int], int] = {}
        da: dict[T, int] = {}
        len1 = len(s1)
        len2 = len(s2)
        maxdist = len1 + len2
        d[-1, -1] = maxdist
        for i in range(len(s1) + 1):
            d[i, -1] = maxdist
            d[i, 0] = i
        for j in range(len(s2) + 1):
            d[-1, j] = maxdist
            d[0, j] = j
        for (i, cs1) in enumerate(s1, start=1):
            db = 0
            for (j, cs2) in enumerate(s2, start=1):
                i1 = da.get(cs2, 0)
                j1 = db
                if self.test_func(cs1, cs2):
                    cost = 0
                    db = j
                else:
                    cost = 1
                d[i, j] = min(d[i - 1, j - 1] + cost, d[i, j - 1] + 1, d[i - 1, j] + 1, d[i1 - 1, j1 - 1] + (i - i1) - 1 + (j - j1))
            da[cs1] = i
        return d[len1, len2]

    def _pure_python_restricted(self, s1: Sequence[T], s2: Sequence[T]) -> int:
        if False:
            while True:
                i = 10
        '\n        https://www.guyrutenberg.com/2008/12/15/damerau-levenshtein-distance-in-python/\n        '
        d: dict[tuple[int, int], int] = {}
        for i in range(-1, len(s1) + 1):
            d[i, -1] = i + 1
        for j in range(-1, len(s2) + 1):
            d[-1, j] = j + 1
        for (i, cs1) in enumerate(s1):
            for (j, cs2) in enumerate(s2):
                cost = int(not self.test_func(cs1, cs2))
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
                if not i or not j:
                    continue
                if not self.test_func(cs1, s2[j - 1]):
                    continue
                if not self.test_func(s1[i - 1], cs2):
                    continue
                d[i, j] = min(d[i, j], d[i - 2, j - 2] + cost)
        return d[len(s1) - 1, len(s2) - 1]

    def __call__(self, s1: Sequence[T], s2: Sequence[T]) -> int:
        if False:
            print('Hello World!')
        (s1, s2) = self._get_sequences(s1, s2)
        result = self.quick_answer(s1, s2)
        if result is not None:
            return result
        if self.restricted:
            return self._pure_python_restricted(s1, s2)
        return self._pure_python_unrestricted(s1, s2)

class JaroWinkler(_BaseSimilarity):
    """
    Computes the Jaro-Winkler measure between two strings.
    The Jaro-Winkler measure is designed to capture cases where two strings
    have a low Jaro score, but share a prefix.
    and thus are likely to match.

    https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance
    https://github.com/Yomguithereal/talisman/blob/master/src/metrics/jaro.js
    https://github.com/Yomguithereal/talisman/blob/master/src/metrics/jaro-winkler.js
    """

    def __init__(self, long_tolerance: bool=False, winklerize: bool=True, qval: int=1, external: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.qval = qval
        self.long_tolerance = long_tolerance
        self.winklerize = winklerize
        self.external = external

    def maximum(self, *sequences: Sequence[object]) -> int:
        if False:
            print('Hello World!')
        return 1

    def __call__(self, s1: Sequence[T], s2: Sequence[T], prefix_weight: float=0.1) -> float:
        if False:
            while True:
                i = 10
        (s1, s2) = self._get_sequences(s1, s2)
        result = self.quick_answer(s1, s2)
        if result is not None:
            return result
        s1_len = len(s1)
        s2_len = len(s2)
        if not s1_len or not s2_len:
            return 0.0
        min_len = min(s1_len, s2_len)
        search_range = max(s1_len, s2_len)
        search_range = search_range // 2 - 1
        if search_range < 0:
            search_range = 0
        s1_flags = [False] * s1_len
        s2_flags = [False] * s2_len
        common_chars = 0
        for (i, s1_ch) in enumerate(s1):
            low = max(0, i - search_range)
            hi = min(i + search_range, s2_len - 1)
            for j in range(low, hi + 1):
                if not s2_flags[j] and s2[j] == s1_ch:
                    s1_flags[i] = s2_flags[j] = True
                    common_chars += 1
                    break
        if not common_chars:
            return 0.0
        k = trans_count = 0
        for (i, s1_f) in enumerate(s1_flags):
            if s1_f:
                for j in range(k, s2_len):
                    if s2_flags[j]:
                        k = j + 1
                        break
                if s1[i] != s2[j]:
                    trans_count += 1
        trans_count //= 2
        weight = common_chars / s1_len + common_chars / s2_len
        weight += (common_chars - trans_count) / common_chars
        weight /= 3
        if not self.winklerize:
            return weight
        if weight <= 0.7:
            return weight
        j = min(min_len, 4)
        i = 0
        while i < j and s1[i] == s2[i]:
            i += 1
        if i:
            weight += i * prefix_weight * (1.0 - weight)
        if not self.long_tolerance or min_len <= 4:
            return weight
        if common_chars <= i + 1 or 2 * common_chars < min_len + i:
            return weight
        tmp = (common_chars - i - 1) / (s1_len + s2_len - i * 2 + 2)
        weight += (1.0 - weight) * tmp
        return weight

class Jaro(JaroWinkler):

    def __init__(self, long_tolerance: bool=False, qval: int=1, external: bool=True) -> None:
        if False:
            while True:
                i = 10
        super().__init__(long_tolerance=long_tolerance, winklerize=False, qval=qval, external=external)

class NeedlemanWunsch(_BaseSimilarity):
    """
    Computes the Needleman-Wunsch measure between two strings.
    The Needleman-Wunsch generalizes the Levenshtein distance and considers global
    alignment between two strings. Specifically, it is computed by assigning
    a score to each alignment between two input strings and choosing the
    score of the best alignment, that is, the maximal score.
    An alignment between two strings is a set of correspondences between the
    characters of between them, allowing for gaps.

    https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
    """

    def __init__(self, gap_cost: float=1.0, sim_func: SimFunc=None, qval: int=1, external: bool=True) -> None:
        if False:
            print('Hello World!')
        self.qval = qval
        self.gap_cost = gap_cost
        if sim_func:
            self.sim_func = sim_func
        else:
            self.sim_func = self._ident
        self.external = external

    def minimum(self, *sequences: Sequence[object]) -> float:
        if False:
            i = 10
            return i + 15
        return -max(map(len, sequences)) * self.gap_cost

    def maximum(self, *sequences: Sequence[object]) -> float:
        if False:
            return 10
        return max(map(len, sequences))

    def distance(self, *sequences: Sequence[object]) -> float:
        if False:
            return 10
        'Get distance between sequences\n        '
        return -1 * self.similarity(*sequences)

    def normalized_distance(self, *sequences: Sequence[object]) -> float:
        if False:
            return 10
        'Get distance from 0 to 1\n        '
        minimum = self.minimum(*sequences)
        maximum = self.maximum(*sequences)
        if maximum == 0:
            return 0
        return (self.distance(*sequences) - minimum) / (maximum - minimum)

    def normalized_similarity(self, *sequences: Sequence[object]) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Get similarity from 0 to 1\n        '
        minimum = self.minimum(*sequences)
        maximum = self.maximum(*sequences)
        if maximum == 0:
            return 1
        return (self.similarity(*sequences) - minimum) / (maximum * 2)

    def __call__(self, s1: Sequence[T], s2: Sequence[T]) -> float:
        if False:
            i = 10
            return i + 15
        if not numpy:
            raise ImportError('Please, install numpy for Needleman-Wunsch measure')
        (s1, s2) = self._get_sequences(s1, s2)
        dist_mat = numpy.zeros((len(s1) + 1, len(s2) + 1), dtype=float)
        for i in range(len(s1) + 1):
            dist_mat[i, 0] = -(i * self.gap_cost)
        for j in range(len(s2) + 1):
            dist_mat[0, j] = -(j * self.gap_cost)
        for (i, c1) in enumerate(s1, 1):
            for (j, c2) in enumerate(s2, 1):
                match = dist_mat[i - 1, j - 1] + self.sim_func(c1, c2)
                delete = dist_mat[i - 1, j] - self.gap_cost
                insert = dist_mat[i, j - 1] - self.gap_cost
                dist_mat[i, j] = max(match, delete, insert)
        return dist_mat[dist_mat.shape[0] - 1, dist_mat.shape[1] - 1]

class SmithWaterman(_BaseSimilarity):
    """
    Computes the Smith-Waterman measure between two strings.
    The Smith-Waterman algorithm performs local sequence alignment;
    that is, for determining similar regions between two strings.
    Instead of looking at the total sequence, the Smith-Waterman algorithm compares
    segments of all possible lengths and optimizes the similarity measure.

    https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm
    https://github.com/Yomguithereal/talisman/blob/master/src/metrics/smith-waterman.js
    """

    def __init__(self, gap_cost: float=1.0, sim_func: SimFunc=None, qval: int=1, external: bool=True) -> None:
        if False:
            return 10
        self.qval = qval
        self.gap_cost = gap_cost
        self.sim_func = sim_func or self._ident
        self.external = external

    def maximum(self, *sequences: Sequence[object]) -> int:
        if False:
            return 10
        return min(map(len, sequences))

    def __call__(self, s1: Sequence[T], s2: Sequence[T]) -> float:
        if False:
            while True:
                i = 10
        if not numpy:
            raise ImportError('Please, install numpy for Smith-Waterman measure')
        (s1, s2) = self._get_sequences(s1, s2)
        result = self.quick_answer(s1, s2)
        if result is not None:
            return result
        dist_mat = numpy.zeros((len(s1) + 1, len(s2) + 1), dtype=float)
        for (i, sc1) in enumerate(s1, start=1):
            for (j, sc2) in enumerate(s2, start=1):
                match = dist_mat[i - 1, j - 1] + self.sim_func(sc1, sc2)
                delete = dist_mat[i - 1, j] - self.gap_cost
                insert = dist_mat[i, j - 1] - self.gap_cost
                dist_mat[i, j] = max(0, match, delete, insert)
        return dist_mat[dist_mat.shape[0] - 1, dist_mat.shape[1] - 1]

class Gotoh(NeedlemanWunsch):
    """Gotoh score
    Gotoh's algorithm is essentially Needleman-Wunsch with affine gap
    penalties:
    https://www.cs.umd.edu/class/spring2003/cmsc838t/papers/gotoh1982.pdf
    """

    def __init__(self, gap_open: int=1, gap_ext: float=0.4, sim_func: SimFunc=None, qval: int=1, external: bool=True) -> None:
        if False:
            return 10
        self.qval = qval
        self.gap_open = gap_open
        self.gap_ext = gap_ext
        if sim_func:
            self.sim_func = sim_func
        else:
            self.sim_func = self._ident
        self.external = external

    def minimum(self, *sequences: Sequence[object]) -> int:
        if False:
            print('Hello World!')
        return -min(map(len, sequences))

    def maximum(self, *sequences: Sequence[object]) -> int:
        if False:
            return 10
        return min(map(len, sequences))

    def __call__(self, s1: Sequence[T], s2: Sequence[T]) -> float:
        if False:
            print('Hello World!')
        if not numpy:
            raise ImportError('Please, install numpy for Gotoh measure')
        (s1, s2) = self._get_sequences(s1, s2)
        len_s1 = len(s1)
        len_s2 = len(s2)
        d_mat = numpy.zeros((len_s1 + 1, len_s2 + 1), dtype=float)
        p_mat = numpy.zeros((len_s1 + 1, len_s2 + 1), dtype=float)
        q_mat = numpy.zeros((len_s1 + 1, len_s2 + 1), dtype=float)
        d_mat[0, 0] = 0
        p_mat[0, 0] = float('-inf')
        q_mat[0, 0] = float('-inf')
        for i in range(1, len_s1 + 1):
            d_mat[i, 0] = float('-inf')
            p_mat[i, 0] = -self.gap_open - self.gap_ext * (i - 1)
            q_mat[i, 0] = float('-inf')
            q_mat[i, 1] = -self.gap_open
        for j in range(1, len_s2 + 1):
            d_mat[0, j] = float('-inf')
            p_mat[0, j] = float('-inf')
            p_mat[1, j] = -self.gap_open
            q_mat[0, j] = -self.gap_open - self.gap_ext * (j - 1)
        for (i, sc1) in enumerate(s1, start=1):
            for (j, sc2) in enumerate(s2, start=1):
                sim_val = self.sim_func(sc1, sc2)
                d_mat[i, j] = max(d_mat[i - 1, j - 1] + sim_val, p_mat[i - 1, j - 1] + sim_val, q_mat[i - 1, j - 1] + sim_val)
                p_mat[i, j] = max(d_mat[i - 1, j] - self.gap_open, p_mat[i - 1, j] - self.gap_ext)
                q_mat[i, j] = max(d_mat[i, j - 1] - self.gap_open, q_mat[i, j - 1] - self.gap_ext)
        (i, j) = (n - 1 for n in d_mat.shape)
        return max(d_mat[i, j], p_mat[i, j], q_mat[i, j])

class StrCmp95(_BaseSimilarity):
    """strcmp95 similarity

    http://cpansearch.perl.org/src/SCW/Text-JaroWinkler-0.1/strcmp95.c
    """
    sp_mx: tuple[tuple[str, str], ...] = (('A', 'E'), ('A', 'I'), ('A', 'O'), ('A', 'U'), ('B', 'V'), ('E', 'I'), ('E', 'O'), ('E', 'U'), ('I', 'O'), ('I', 'U'), ('O', 'U'), ('I', 'Y'), ('E', 'Y'), ('C', 'G'), ('E', 'F'), ('W', 'U'), ('W', 'V'), ('X', 'K'), ('S', 'Z'), ('X', 'S'), ('Q', 'C'), ('U', 'V'), ('M', 'N'), ('L', 'I'), ('Q', 'O'), ('P', 'R'), ('I', 'J'), ('2', 'Z'), ('5', 'S'), ('8', 'B'), ('1', 'I'), ('1', 'L'), ('0', 'O'), ('0', 'Q'), ('C', 'K'), ('G', 'J'))

    def __init__(self, long_strings: bool=False, external: bool=True) -> None:
        if False:
            return 10
        self.long_strings = long_strings
        self.external = external

    def maximum(self, *sequences: Sequence[object]) -> int:
        if False:
            while True:
                i = 10
        return 1

    @staticmethod
    def _in_range(char) -> bool:
        if False:
            return 10
        return 0 < ord(char) < 91

    def __call__(self, s1: str, s2: str) -> float:
        if False:
            return 10
        s1 = s1.strip().upper()
        s2 = s2.strip().upper()
        result = self.quick_answer(s1, s2)
        if result is not None:
            return result
        len_s1 = len(s1)
        len_s2 = len(s2)
        adjwt = defaultdict(int)
        for (c1, c2) in self.sp_mx:
            adjwt[c1, c2] = 3
            adjwt[c2, c1] = 3
        if len_s1 > len_s2:
            search_range = len_s1
            minv = len_s2
        else:
            search_range = len_s2
            minv = len_s1
        s1_flag = [0] * search_range
        s2_flag = [0] * search_range
        search_range = max(0, search_range // 2 - 1)
        num_com = 0
        yl1 = len_s2 - 1
        for (i, sc1) in enumerate(s1):
            lowlim = max(i - search_range, 0)
            hilim = min(i + search_range, yl1)
            for j in range(lowlim, hilim + 1):
                if s2_flag[j] == 0 and s2[j] == sc1:
                    s2_flag[j] = 1
                    s1_flag[i] = 1
                    num_com += 1
                    break
        if num_com == 0:
            return 0.0
        k = n_trans = 0
        for (i, sc1) in enumerate(s1):
            if not s1_flag[i]:
                continue
            for j in range(k, len_s2):
                if s2_flag[j] != 0:
                    k = j + 1
                    break
            if sc1 != s2[j]:
                n_trans += 1
        n_trans = n_trans // 2
        n_simi = 0
        if minv > num_com:
            for i in range(len_s1):
                if s1_flag[i] != 0:
                    continue
                if not self._in_range(s1[i]):
                    continue
                for j in range(len_s2):
                    if s2_flag[j] != 0:
                        continue
                    if not self._in_range(s2[j]):
                        continue
                    if (s1[i], s2[j]) not in adjwt:
                        continue
                    n_simi += adjwt[s1[i], s2[j]]
                    s2_flag[j] = 2
                    break
        num_sim = n_simi / 10.0 + num_com
        weight = num_sim / len_s1 + num_sim / len_s2
        weight += (num_com - n_trans) / num_com
        weight = weight / 3.0
        if weight <= 0.7:
            return weight
        j = min(minv, 4)
        i = 0
        for (sc1, sc2) in zip(s1, s2):
            if i >= j:
                break
            if sc1 != sc2:
                break
            if sc1.isdigit():
                break
            i += 1
        if i:
            weight += i * 0.1 * (1.0 - weight)
        if not self.long_strings:
            return weight
        if minv <= 4:
            return weight
        if num_com <= i + 1 or 2 * num_com < minv + i:
            return weight
        if s1[0].isdigit():
            return weight
        res = (num_com - i - 1) / (len_s1 + len_s2 - i * 2 + 2)
        weight += (1.0 - weight) * res
        return weight

class MLIPNS(_BaseSimilarity):
    """
    Compute the Hamming distance between the two or more sequences.
    The Hamming distance is the number of differing items in ordered sequences.

    http://www.sial.iias.spb.su/files/386-386-1-PB.pdf
    https://github.com/Yomguithereal/talisman/blob/master/src/metrics/mlipns.js
    """

    def __init__(self, threshold: float=0.25, maxmismatches: int=2, qval: int=1, external: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        self.qval = qval
        self.threshold = threshold
        self.maxmismatches = maxmismatches
        self.external = external

    def maximum(self, *sequences: Sequence[object]) -> int:
        if False:
            print('Hello World!')
        return 1

    def __call__(self, *sequences: Sequence[object]) -> float:
        if False:
            while True:
                i = 10
        sequences = self._get_sequences(*sequences)
        result = self.quick_answer(*sequences)
        if result is not None:
            return result
        mismatches = 0
        ham = Hamming()(*sequences)
        maxlen = max(map(len, sequences))
        while all(sequences) and mismatches <= self.maxmismatches:
            if not maxlen:
                return 1
            if 1 - (maxlen - ham) / maxlen <= self.threshold:
                return 1
            mismatches += 1
            ham -= 1
            maxlen -= 1
        if not maxlen:
            return 1
        return 0
hamming = Hamming()
levenshtein = Levenshtein()
damerau = damerau_levenshtein = DamerauLevenshtein()
jaro = Jaro()
jaro_winkler = JaroWinkler()
needleman_wunsch = NeedlemanWunsch()
smith_waterman = SmithWaterman()
gotoh = Gotoh()
strcmp95 = StrCmp95()
mlipns = MLIPNS()