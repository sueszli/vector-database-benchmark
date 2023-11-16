class Solution(object):

    def getMinDistance(self, nums, target, start):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type target: int\n        :type start: int\n        :rtype: int\n        '
        for i in xrange(len(nums)):
            if start - i >= 0 and nums[start - i] == target or (start + i < len(nums) and nums[start + i] == target):
                break
        return i