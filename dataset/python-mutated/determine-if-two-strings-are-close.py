import collections

class Solution(object):

    def closeStrings(self, word1, word2):
        if False:
            return 10
        '\n        :type word1: str\n        :type word2: str\n        :rtype: bool\n        '
        if len(word1) != len(word2):
            return False
        (cnt1, cnt2) = (collections.Counter(word1), collections.Counter(word2))
        return set(cnt1.iterkeys()) == set(cnt2.iterkeys()) and collections.Counter(cnt1.itervalues()) == collections.Counter(cnt2.itervalues())