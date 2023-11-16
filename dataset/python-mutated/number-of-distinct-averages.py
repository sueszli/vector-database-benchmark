class Solution(object):

    def distinctAverages(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        lookup = set()
        nums.sort()
        (left, right) = (0, len(nums) - 1)
        while left < right:
            lookup.add(nums[left] + nums[right])
            (left, right) = (left + 1, right - 1)
        return len(lookup)