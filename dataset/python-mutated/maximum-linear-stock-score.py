import collections

class Solution(object):

    def maxScore(self, prices):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type prices: List[int]\n        :rtype: int\n        '
        cnt = collections.Counter()
        for (i, x) in enumerate(prices):
            cnt[x - i] += x
        return max(cnt.itervalues())