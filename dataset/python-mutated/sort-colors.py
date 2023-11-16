class Solution(object):

    def sortColors(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: void Do not return anything, modify nums in-place instead.\n        '

        def triPartition(nums, target):
            if False:
                while True:
                    i = 10
            (i, left, right) = (0, 0, len(nums) - 1)
            while i <= right:
                if nums[i] > target:
                    (nums[i], nums[right]) = (nums[right], nums[i])
                    right -= 1
                else:
                    if nums[i] < target:
                        (nums[left], nums[i]) = (nums[i], nums[left])
                        left += 1
                    i += 1
        triPartition(nums, 1)