import collections

class Solution(object):

    def findTargetSumWays(self, nums, S):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type S: int\n        :rtype: int\n        '

        def subsetSum(nums, S):
            if False:
                print('Hello World!')
            dp = collections.defaultdict(int)
            dp[0] = 1
            for n in nums:
                for i in reversed(xrange(n, S + 1)):
                    if i - n in dp:
                        dp[i] += dp[i - n]
            return dp[S]
        total = sum(nums)
        if total < S or (S + total) % 2:
            return 0
        P = (S + total) // 2
        return subsetSum(nums, P)