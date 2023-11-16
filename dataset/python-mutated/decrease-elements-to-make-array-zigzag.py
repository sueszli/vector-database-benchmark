class Solution(object):

    def movesToMakeZigzag(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = [0, 0]
        for i in xrange(len(nums)):
            left = nums[i - 1] if i - 1 >= 0 else float('inf')
            right = nums[i + 1] if i + 1 < len(nums) else float('inf')
            result[i % 2] += max(nums[i] - min(left, right) + 1, 0)
        return min(result)