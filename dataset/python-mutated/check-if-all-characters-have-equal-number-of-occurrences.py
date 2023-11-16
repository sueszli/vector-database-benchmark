import collections

class Solution(object):

    def areOccurrencesEqual(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: bool\n        '
        return len(set(collections.Counter(s).itervalues())) == 1