class Solution(object):

    def countSubarrays(self, nums, minK, maxK):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type minK: int\n        :type maxK: int\n        :rtype: int\n        '
        result = left = 0
        right = [-1] * 2
        for (i, x) in enumerate(nums):
            if not minK <= x <= maxK:
                left = i + 1
                continue
            if x == minK:
                right[0] = i
            if x == maxK:
                right[1] = i
            result += max(min(right) - left + 1, 0)
        return result