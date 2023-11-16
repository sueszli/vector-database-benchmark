class Solution:

    def selectionSort(self, nums: [int]) -> [int]:
        if False:
            while True:
                i = 10
        for i in range(len(nums) - 1):
            min_i = i
            for j in range(i + 1, len(nums)):
                if nums[j] < nums[min_i]:
                    min_i = j
            if i != min_i:
                (nums[i], nums[min_i]) = (nums[min_i], nums[i])
        return nums

    def sortArray(self, nums: [int]) -> [int]:
        if False:
            return 10
        return self.selectionSort(nums)
print(Solution().sortArray([5, 2, 3, 6, 1, 4]))