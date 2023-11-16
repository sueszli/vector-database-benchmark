class Solution:

    def removeDuplicates(self, nums):
        if False:
            for i in range(10):
                print('nop')
        if len(nums) == 0:
            return 0
        c = 1
        for i in range(1, len(nums)):
            if nums[i] > nums[c - 1]:
                nums[c] = nums[i]
                c += 1
        return c