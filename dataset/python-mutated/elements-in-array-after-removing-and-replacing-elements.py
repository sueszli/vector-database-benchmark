class Solution(object):

    def elementInNums(self, nums, queries):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type queries: List[List[int]]\n        :rtype: List[int]\n        '
        result = []
        for (t, i) in queries:
            t %= 2 * len(nums)
            if t + i < len(nums):
                result.append(nums[t + i])
            elif i < t - len(nums):
                result.append(nums[i])
            else:
                result.append(-1)
        return result