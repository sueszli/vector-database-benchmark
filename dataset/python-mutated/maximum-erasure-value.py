class Solution(object):

    def maximumUniqueSubarray(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        lookup = {}
        prefix = [0] * (len(nums) + 1)
        (result, left) = (0, 0)
        for (right, num) in enumerate(nums):
            prefix[right + 1] = prefix[right] + num
            if num in lookup:
                left = max(left, lookup[num] + 1)
            lookup[num] = right
            result = max(result, prefix[right + 1] - prefix[left])
        return result