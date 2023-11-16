class Solution:

    def merge(self, left_nums: [int], right_nums: [int]):
        if False:
            print('Hello World!')
        nums = []
        (left_i, right_i) = (0, 0)
        while left_i < len(left_nums) and right_i < len(right_nums):
            if left_nums[left_i] < right_nums[right_i]:
                nums.append(left_nums[left_i])
                left_i += 1
            else:
                nums.append(right_nums[right_i])
                right_i += 1
        while left_i < len(left_nums):
            nums.append(left_nums[left_i])
            left_i += 1
        while right_i < len(right_nums):
            nums.append(right_nums[right_i])
            right_i += 1
        return nums

    def mergeSort(self, nums: [int]) -> [int]:
        if False:
            return 10
        if len(nums) <= 1:
            return nums
        mid = len(nums) // 2
        left_nums = self.mergeSort(nums[0:mid])
        right_nums = self.mergeSort(nums[mid:])
        return self.merge(left_nums, right_nums)

    def sortArray(self, nums: [int]) -> [int]:
        if False:
            while True:
                i = 10
        return self.mergeSort(nums)
print(Solution().sortArray([0, 5, 7, 3, 1, 6, 8, 4]))