class Solution(object):

    def maxDivScore(self, nums, divisors):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type divisors: List[int]\n        :rtype: int\n        '
        return max(divisors, key=lambda d: (sum((x % d == 0 for x in nums)), -d))