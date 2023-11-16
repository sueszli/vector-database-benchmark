import collections

class Solution(object):

    def minimumKeypresses(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: int\n        '
        return sum((cnt * (i // 9 + 1) for (i, cnt) in enumerate(sorted(collections.Counter(s).itervalues(), reverse=True))))