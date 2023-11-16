class Solution(object):

    def minIncrementOperations(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        W = 3
        dp = [0] * W
        for (i, x) in enumerate(nums):
            dp[i % W] = min((dp[j % W] for j in xrange(i - W, i))) + max(k - x, 0)
        return min(dp)