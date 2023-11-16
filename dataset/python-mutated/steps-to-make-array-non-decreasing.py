class Solution(object):

    def totalSteps(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        dp = [0] * len(nums)
        stk = []
        for i in reversed(xrange(len(nums))):
            while stk and nums[stk[-1]] < nums[i]:
                dp[i] = max(dp[i] + 1, dp[stk.pop()])
            stk.append(i)
        return max(dp)

class Solution2(object):

    def totalSteps(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        dp = [0] * len(nums)
        stk = []
        for i in xrange(len(nums)):
            curr = 0
            while stk and nums[stk[-1]] <= nums[i]:
                curr = max(curr, dp[stk.pop()])
            if stk:
                dp[i] = curr + 1
            stk.append(i)
        return max(dp)