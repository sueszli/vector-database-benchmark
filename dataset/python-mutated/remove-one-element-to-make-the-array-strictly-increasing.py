class Solution(object):

    def canBeIncreasing(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: bool\n        '
        deleted = False
        for i in xrange(1, len(nums)):
            if nums[i] > nums[i - 1]:
                continue
            if deleted:
                return False
            deleted = True
            if i >= 2 and nums[i - 2] > nums[i]:
                nums[i] = nums[i - 1]
        return True