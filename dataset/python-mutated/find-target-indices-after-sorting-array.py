class Solution(object):

    def targetIndices(self, nums, target):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: List[int]\n        '
        less = sum((x < target for x in nums))
        return range(less, less + sum((x == target for x in nums)))