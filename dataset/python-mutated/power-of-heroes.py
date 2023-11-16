class Solution(object):

    def sumOfPower(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        nums.sort()
        result = dp = 0
        for x in nums:
            result = (result + x ** 2 * (dp + x)) % MOD
            dp = (dp + (dp + x)) % MOD
        return result