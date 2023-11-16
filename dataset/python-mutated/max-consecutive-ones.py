class Solution(object):

    def findMaxConsecutiveOnes(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (result, local_max) = (0, 0)
        for n in nums:
            local_max = local_max + 1 if n else 0
            result = max(result, local_max)
        return result