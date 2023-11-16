from random import randint

class Solution(object):

    def minMoves2(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def kthElement(nums, k):
            if False:
                print('Hello World!')

            def PartitionAroundPivot(left, right, pivot_idx, nums):
                if False:
                    print('Hello World!')
                pivot_value = nums[pivot_idx]
                new_pivot_idx = left
                (nums[pivot_idx], nums[right]) = (nums[right], nums[pivot_idx])
                for i in xrange(left, right):
                    if nums[i] > pivot_value:
                        (nums[i], nums[new_pivot_idx]) = (nums[new_pivot_idx], nums[i])
                        new_pivot_idx += 1
                (nums[right], nums[new_pivot_idx]) = (nums[new_pivot_idx], nums[right])
                return new_pivot_idx
            (left, right) = (0, len(nums) - 1)
            while left <= right:
                pivot_idx = randint(left, right)
                new_pivot_idx = PartitionAroundPivot(left, right, pivot_idx, nums)
                if new_pivot_idx == k:
                    return nums[new_pivot_idx]
                elif new_pivot_idx > k:
                    right = new_pivot_idx - 1
                else:
                    left = new_pivot_idx + 1
        median = kthElement(nums, len(nums) // 2)
        return sum((abs(num - median) for num in nums))

    def minMoves22(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        median = sorted(nums)[len(nums) / 2]
        return sum((abs(num - median) for num in nums))