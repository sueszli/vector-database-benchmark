import collections

class Solution(object):

    def unequalTriplets(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        K = 3
        cnt = collections.Counter()
        dp = [0] * K
        for x in nums:
            cnt[x] += 1
            other_cnt = 1
            for i in xrange(K):
                dp[i] += other_cnt
                other_cnt = dp[i] - cnt[x] * other_cnt
        return dp[K - 1]