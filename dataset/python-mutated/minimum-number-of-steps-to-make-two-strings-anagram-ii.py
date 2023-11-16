import collections

class Solution(object):

    def minSteps(self, s, t):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type t: str\n        :rtype: int\n        '
        (cnt1, cnt2) = (collections.Counter(s), collections.Counter(t))
        return sum((cnt1 - cnt2).itervalues()) + sum((cnt2 - cnt1).itervalues())