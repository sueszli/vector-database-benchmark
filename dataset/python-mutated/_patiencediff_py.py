from __future__ import absolute_import
from bisect import bisect
import difflib
from bzrlib.trace import mutter
__all__ = ['PatienceSequenceMatcher', 'unified_diff', 'unified_diff_files']

def unique_lcs_py(a, b):
    if False:
        while True:
            i = 10
    'Find the longest common subset for unique lines.\n\n    :param a: An indexable object (such as string or list of strings)\n    :param b: Another indexable object (such as string or list of strings)\n    :return: A list of tuples, one for each line which is matched.\n            [(line_in_a, line_in_b), ...]\n\n    This only matches lines which are unique on both sides.\n    This helps prevent common lines from over influencing match\n    results.\n    The longest common subset uses the Patience Sorting algorithm:\n    http://en.wikipedia.org/wiki/Patience_sorting\n    '
    index = {}
    for i in xrange(len(a)):
        line = a[i]
        if line in index:
            index[line] = None
        else:
            index[line] = i
    btoa = [None] * len(b)
    index2 = {}
    for (pos, line) in enumerate(b):
        next = index.get(line)
        if next is not None:
            if line in index2:
                btoa[index2[line]] = None
                del index[line]
            else:
                index2[line] = pos
                btoa[pos] = next
    backpointers = [None] * len(b)
    stacks = []
    lasts = []
    k = 0
    for (bpos, apos) in enumerate(btoa):
        if apos is None:
            continue
        if stacks and stacks[-1] < apos:
            k = len(stacks)
        elif stacks and stacks[k] < apos and (k == len(stacks) - 1 or stacks[k + 1] > apos):
            k += 1
        else:
            k = bisect(stacks, apos)
        if k > 0:
            backpointers[bpos] = lasts[k - 1]
        if k < len(stacks):
            stacks[k] = apos
            lasts[k] = bpos
        else:
            stacks.append(apos)
            lasts.append(bpos)
    if len(lasts) == 0:
        return []
    result = []
    k = lasts[-1]
    while k is not None:
        result.append((btoa[k], k))
        k = backpointers[k]
    result.reverse()
    return result

def recurse_matches_py(a, b, alo, blo, ahi, bhi, answer, maxrecursion):
    if False:
        for i in range(10):
            print('nop')
    'Find all of the matching text in the lines of a and b.\n\n    :param a: A sequence\n    :param b: Another sequence\n    :param alo: The start location of a to check, typically 0\n    :param ahi: The start location of b to check, typically 0\n    :param ahi: The maximum length of a to check, typically len(a)\n    :param bhi: The maximum length of b to check, typically len(b)\n    :param answer: The return array. Will be filled with tuples\n                   indicating [(line_in_a, line_in_b)]\n    :param maxrecursion: The maximum depth to recurse.\n                         Must be a positive integer.\n    :return: None, the return value is in the parameter answer, which\n             should be a list\n\n    '
    if maxrecursion < 0:
        mutter('max recursion depth reached')
        return
    oldlength = len(answer)
    if alo == ahi or blo == bhi:
        return
    last_a_pos = alo - 1
    last_b_pos = blo - 1
    for (apos, bpos) in unique_lcs_py(a[alo:ahi], b[blo:bhi]):
        apos += alo
        bpos += blo
        if last_a_pos + 1 != apos or last_b_pos + 1 != bpos:
            recurse_matches_py(a, b, last_a_pos + 1, last_b_pos + 1, apos, bpos, answer, maxrecursion - 1)
        last_a_pos = apos
        last_b_pos = bpos
        answer.append((apos, bpos))
    if len(answer) > oldlength:
        recurse_matches_py(a, b, last_a_pos + 1, last_b_pos + 1, ahi, bhi, answer, maxrecursion - 1)
    elif a[alo] == b[blo]:
        while alo < ahi and blo < bhi and (a[alo] == b[blo]):
            answer.append((alo, blo))
            alo += 1
            blo += 1
        recurse_matches_py(a, b, alo, blo, ahi, bhi, answer, maxrecursion - 1)
    elif a[ahi - 1] == b[bhi - 1]:
        nahi = ahi - 1
        nbhi = bhi - 1
        while nahi > alo and nbhi > blo and (a[nahi - 1] == b[nbhi - 1]):
            nahi -= 1
            nbhi -= 1
        recurse_matches_py(a, b, last_a_pos + 1, last_b_pos + 1, nahi, nbhi, answer, maxrecursion - 1)
        for i in xrange(ahi - nahi):
            answer.append((nahi + i, nbhi + i))

def _collapse_sequences(matches):
    if False:
        print('Hello World!')
    'Find sequences of lines.\n\n    Given a sequence of [(line_in_a, line_in_b),]\n    find regions where they both increment at the same time\n    '
    answer = []
    start_a = start_b = None
    length = 0
    for (i_a, i_b) in matches:
        if start_a is not None and i_a == start_a + length and (i_b == start_b + length):
            length += 1
        else:
            if start_a is not None:
                answer.append((start_a, start_b, length))
            start_a = i_a
            start_b = i_b
            length = 1
    if length != 0:
        answer.append((start_a, start_b, length))
    return answer

def _check_consistency(answer):
    if False:
        while True:
            i = 10
    next_a = -1
    next_b = -1
    for (a, b, match_len) in answer:
        if a < next_a:
            raise ValueError('Non increasing matches for a')
        if b < next_b:
            raise ValueError('Non increasing matches for b')
        next_a = a + match_len
        next_b = b + match_len

class PatienceSequenceMatcher_py(difflib.SequenceMatcher):
    """Compare a pair of sequences using longest common subset."""
    _do_check_consistency = True

    def __init__(self, isjunk=None, a='', b=''):
        if False:
            print('Hello World!')
        if isjunk is not None:
            raise NotImplementedError('Currently we do not support isjunk for sequence matching')
        difflib.SequenceMatcher.__init__(self, isjunk, a, b)

    def get_matching_blocks(self):
        if False:
            for i in range(10):
                print('nop')
        'Return list of triples describing matching subsequences.\n\n        Each triple is of the form (i, j, n), and means that\n        a[i:i+n] == b[j:j+n].  The triples are monotonically increasing in\n        i and in j.\n\n        The last triple is a dummy, (len(a), len(b), 0), and is the only\n        triple with n==0.\n\n        >>> s = PatienceSequenceMatcher(None, "abxcd", "abcd")\n        >>> s.get_matching_blocks()\n        [(0, 0, 2), (3, 2, 2), (5, 4, 0)]\n        '
        if self.matching_blocks is not None:
            return self.matching_blocks
        matches = []
        recurse_matches_py(self.a, self.b, 0, 0, len(self.a), len(self.b), matches, 10)
        self.matching_blocks = _collapse_sequences(matches)
        self.matching_blocks.append((len(self.a), len(self.b), 0))
        if PatienceSequenceMatcher_py._do_check_consistency:
            if __debug__:
                _check_consistency(self.matching_blocks)
        return self.matching_blocks