import collections
import itertools

class Solution(object):

    def similarPairs(self, words):
        if False:
            return 10
        '\n        :type words: List[str]\n        :rtype: int\n        '
        cnt = collections.Counter()
        result = 0
        for w in words:
            mask = reduce(lambda total, x: total | x, itertools.imap(lambda c: 1 << ord(c) - ord('a'), w))
            result += cnt[mask]
            cnt[mask] += 1
        return result