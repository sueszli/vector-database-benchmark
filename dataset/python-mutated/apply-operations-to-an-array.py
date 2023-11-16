class Solution(object):

    def applyOperations(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        for i in xrange(len(nums) - 1):
            if nums[i] == nums[i + 1]:
                (nums[i], nums[i + 1]) = (2 * nums[i], 0)
        i = 0
        for x in nums:
            if not x:
                continue
            nums[i] = x
            i += 1
        for i in xrange(i, len(nums)):
            nums[i] = 0
        return nums