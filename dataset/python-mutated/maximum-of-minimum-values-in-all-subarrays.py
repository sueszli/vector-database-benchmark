class Solution(object):

    def findMaximums(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '

        def find_bound(nums, direction, init):
            if False:
                while True:
                    i = 10
            result = [0] * len(nums)
            stk = [init]
            for i in direction(xrange(len(nums))):
                while stk[-1] != init and nums[stk[-1]] >= nums[i]:
                    stk.pop()
                result[i] = stk[-1]
                stk.append(i)
            return result
        left = find_bound(nums, lambda x: x, -1)
        right = find_bound(nums, reversed, len(nums))
        result = [-1] * len(nums)
        for (i, v) in enumerate(nums):
            result[right[i] - 1 - left[i] - 1] = max(result[right[i] - 1 - left[i] - 1], v)
        for i in reversed(xrange(len(nums) - 1)):
            result[i] = max(result[i], result[i + 1])
        return result