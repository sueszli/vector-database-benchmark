import random

class Solution(object):

    def kthLargestNumber(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[str]\n        :type k: int\n        :rtype: str\n        '

        def nth_element(nums, n, compare=lambda a, b: a < b):
            if False:
                print('Hello World!')

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
        nth_element(nums, k - 1, compare=lambda a, b: a > b if len(a) == len(b) else len(a) > len(b))
        return nums[k - 1]