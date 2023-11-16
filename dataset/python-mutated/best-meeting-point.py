from random import randint

class Solution(object):

    def minTotalDistance(self, grid):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        x = [i for (i, row) in enumerate(grid) for v in row if v == 1]
        y = [j for row in grid for (j, v) in enumerate(row) if v == 1]
        mid_x = self.findKthLargest(x, len(x) / 2 + 1)
        mid_y = self.findKthLargest(y, len(y) / 2 + 1)
        return sum([abs(mid_x - i) + abs(mid_y - j) for (i, row) in enumerate(grid) for (j, v) in enumerate(row) if v == 1])

    def findKthLargest(self, nums, k):
        if False:
            i = 10
            return i + 15
        (left, right) = (0, len(nums) - 1)
        while left <= right:
            pivot_idx = randint(left, right)
            new_pivot_idx = self.PartitionAroundPivot(left, right, pivot_idx, nums)
            if new_pivot_idx == k - 1:
                return nums[new_pivot_idx]
            elif new_pivot_idx > k - 1:
                right = new_pivot_idx - 1
            else:
                left = new_pivot_idx + 1

    def PartitionAroundPivot(self, left, right, pivot_idx, nums):
        if False:
            return 10
        pivot_value = nums[pivot_idx]
        new_pivot_idx = left
        (nums[pivot_idx], nums[right]) = (nums[right], nums[pivot_idx])
        for i in xrange(left, right):
            if nums[i] > pivot_value:
                (nums[i], nums[new_pivot_idx]) = (nums[new_pivot_idx], nums[i])
                new_pivot_idx += 1
        (nums[right], nums[new_pivot_idx]) = (nums[new_pivot_idx], nums[right])
        return new_pivot_idx