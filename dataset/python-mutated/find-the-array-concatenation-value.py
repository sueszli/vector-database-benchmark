import math

class Solution(object):

    def findTheArrayConcVal(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return sum((nums[i] * 10 ** (int(math.log10(nums[~i])) + 1) for i in xrange(len(nums) // 2))) + sum((nums[i] for i in xrange(len(nums) // 2, len(nums))))