class Solution:

    def shellSort(self, nums: [int]) -> [int]:
        if False:
            print('Hello World!')
        size = len(nums)
        gap = size // 2
        while gap > 0:
            for i in range(gap, size):
                temp = nums[i]
                j = i
                while j >= gap and nums[j - gap] > temp:
                    nums[j] = nums[j - gap]
                    j -= gap
                nums[j] = temp
            gap = gap // 2
        return nums

    def sortArray(self, nums: [int]) -> [int]:
        if False:
            print('Hello World!')
        return self.shellSort(nums)
print(Solution().sortArray([7, 2, 6, 8, 0, 4, 1, 5, 9, 3]))