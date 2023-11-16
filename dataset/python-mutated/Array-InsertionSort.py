class Solution:

    def insertionSort(self, nums: [int]) -> [int]:
        if False:
            i = 10
            return i + 15
        for i in range(1, len(nums)):
            temp = nums[i]
            j = i
            while j > 0 and nums[j - 1] > temp:
                nums[j] = nums[j - 1]
                j -= 1
            nums[j] = temp
        return nums

    def sortArray(self, nums: [int]) -> [int]:
        if False:
            i = 10
            return i + 15
        return self.insertionSort(nums)
print(Solution().sortArray([5, 2, 3, 6, 1, 4]))