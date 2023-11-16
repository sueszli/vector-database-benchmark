import collections

class Solution(object):

    def shareCandies(self, candies, k):
        if False:
            while True:
                i = 10
        '\n        :type candies: List[int]\n        :type k: int\n        :rtype: int\n        '
        cnt = collections.Counter((candies[i] for i in xrange(k, len(candies))))
        result = curr = len(cnt)
        for i in xrange(k, len(candies)):
            cnt[candies[i]] -= 1
            curr += (cnt[candies[i - k]] == 0) - (cnt[candies[i]] == 0)
            cnt[candies[i - k]] += 1
            result = max(result, curr)
        return result