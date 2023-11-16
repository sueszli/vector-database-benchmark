class Solution(object):

    def minimumOperations(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        k = 3
        dp = [0] * k
        for x in nums:
            dp[x - 1] += 1
            for i in xrange(x, len(dp)):
                dp[i] = max(dp[i], dp[i - 1])
        return len(nums) - dp[-1]