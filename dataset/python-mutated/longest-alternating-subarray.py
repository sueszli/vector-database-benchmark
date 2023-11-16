class Solution(object):

    def alternatingSubarray(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = l = -1
        for i in xrange(len(nums) - 1):
            if l != -1 and nums[i - 1] == nums[i + 1]:
                l += 1
            else:
                l = 2 if nums[i + 1] - nums[i] == 1 else -1
            result = max(result, l)
        return result