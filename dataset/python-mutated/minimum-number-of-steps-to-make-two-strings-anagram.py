import collections

class Solution(object):

    def minSteps(self, s, t):
        if False:
            return 10
        '\n        :type s: str\n        :type t: str\n        :rtype: int\n        '
        diff = collections.Counter(s) - collections.Counter(t)
        return sum(diff.itervalues())