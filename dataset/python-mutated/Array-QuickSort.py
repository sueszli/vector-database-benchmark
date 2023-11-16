import random

class Solution:

    def randomPartition(self, nums: [int], low: int, high: int) -> int:
        if False:
            i = 10
            return i + 15
        i = random.randint(low, high)
        (nums[i], nums[low]) = (nums[low], nums[i])
        return self.partition(nums, low, high)

    def partition(self, nums: [int], low: int, high: int) -> int:
        if False:
            return 10
        pivot = nums[low]
        (i, j) = (low, high)
        while i < j:
            while i < j and nums[j] >= pivot:
                j -= 1
            while i < j and nums[i] <= pivot:
                i += 1
            (nums[i], nums[j]) = (nums[j], nums[i])
        (nums[j], nums[low]) = (nums[low], nums[j])
        return j

    def quickSort(self, nums: [int], low: int, high: int) -> [int]:
        if False:
            return 10
        if low < high:
            pivot_i = self.partition(nums, low, high)
            self.quickSort(nums, low, pivot_i - 1)
            self.quickSort(nums, pivot_i + 1, high)
        return nums

    def sortArray(self, nums: [int]) -> [int]:
        if False:
            i = 10
            return i + 15
        return self.quickSort(nums, 0, len(nums) - 1)
print(Solution().sortArray([4, 7, 5, 2, 6, 1, 3]))