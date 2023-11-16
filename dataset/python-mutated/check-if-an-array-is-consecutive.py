class Solution(object):

    def isConsecutive(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: bool\n        '
        return max(nums) - min(nums) + 1 == len(nums) == len(set(nums))

class Solution2(object):

    def isConsecutive(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: bool\n        '
        nums.sort()
        return all((nums[i] + 1 == nums[i + 1] for i in xrange(len(nums) - 1)))