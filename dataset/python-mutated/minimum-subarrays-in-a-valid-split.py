class Solution(object):

    def validSubarraySplit(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                while True:
                    i = 10
            while b:
                (a, b) = (b, a % b)
            return a
        dp = [float('inf')] * (len(nums) + 1)
        dp[0] = 0
        for i in xrange(1, len(nums) + 1):
            for j in xrange(i):
                if gcd(nums[j], nums[i - 1]) != 1:
                    dp[i] = min(dp[i], dp[j] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1