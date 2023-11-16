import random

class Solution(object):

    def minOperations(self, grid, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type grid: List[List[int]]\n        :type x: int\n        :rtype: int\n        '

        def nth_element(nums, n, compare=lambda a, b: a < b):
            if False:
                for i in range(10):
                    print('nop')

            def tri_partition(nums, left, right, target, compare):
                if False:
                    while True:
                        i = 10
                mid = left
                while mid <= right:
                    if nums[mid] == target:
                        mid += 1
                    elif compare(nums[mid], target):
                        (nums[left], nums[mid]) = (nums[mid], nums[left])
                        left += 1
                        mid += 1
                    else:
                        (nums[mid], nums[right]) = (nums[right], nums[mid])
                        right -= 1
                return (left, right)
            (left, right) = (0, len(nums) - 1)
            while left <= right:
                pivot_idx = random.randint(left, right)
                (pivot_left, pivot_right) = tri_partition(nums, left, right, nums[pivot_idx], compare)
                if pivot_left <= n <= pivot_right:
                    return
                elif pivot_left > n:
                    right = pivot_left - 1
                else:
                    left = pivot_right + 1
        nums = [v for row in grid for v in row]
        if len(set((v % x for v in nums))) > 1:
            return -1
        nth_element(nums, len(nums) // 2)
        median = nums[len(nums) // 2]
        return sum((abs(v - median) // x for v in nums))