class Solution(object):

    def findMaxConsecutiveOnes(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (result, prev, curr) = (0, 0, 0)
        for n in nums:
            if n == 0:
                result = max(result, prev + curr + 1)
                (prev, curr) = (curr, 0)
            else:
                curr += 1
        return min(max(result, prev + curr + 1), len(nums))