import collections

class Solution(object):

    def rearrangeCharacters(self, s, target):
        if False:
            return 10
        '\n        :type s: str\n        :type target: str\n        :rtype: int\n        '
        cnt1 = collections.Counter(s)
        cnt2 = collections.Counter(target)
        return min((cnt1[k] // v for (k, v) in cnt2.iteritems()))