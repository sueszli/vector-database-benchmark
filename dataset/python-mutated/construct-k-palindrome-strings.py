import collections

class Solution(object):

    def canConstruct(self, s, k):
        if False:
            return 10
        '\n        :type s: str\n        :type k: int\n        :rtype: bool\n        '
        count = collections.Counter(s)
        odd = sum((v % 2 for v in count.itervalues()))
        return odd <= k <= len(s)