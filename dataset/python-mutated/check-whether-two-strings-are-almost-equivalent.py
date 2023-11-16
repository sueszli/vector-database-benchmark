import collections

class Solution(object):

    def checkAlmostEquivalent(self, word1, word2):
        if False:
            return 10
        '\n        :type word1: str\n        :type word2: str\n        :rtype: bool\n        '
        k = 3
        (cnt1, cnt2) = (collections.Counter(word1), collections.Counter(word2))
        return all((abs(cnt1[c] - cnt2[c]) <= k for c in set(cnt1.keys() + cnt2.keys())))