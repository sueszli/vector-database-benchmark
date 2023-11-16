class Solution(object):

    def findSubarrays(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: bool\n        '
        lookup = set()
        for i in xrange(len(nums) - 1):
            if nums[i] + nums[i + 1] in lookup:
                return True
            lookup.add(nums[i] + nums[i + 1])
        return False