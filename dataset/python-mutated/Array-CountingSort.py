class Solution:

    def countingSort(self, nums: [int]) -> [int]:
        if False:
            print('Hello World!')
        (nums_min, nums_max) = (min(nums), max(nums))
        size = nums_max - nums_min + 1
        counts = [0 for _ in range(size)]
        for num in nums:
            counts[num - nums_min] += 1
        for i in range(1, size):
            counts[i] += counts[i - 1]
        res = [0 for _ in range(len(nums))]
        for i in range(len(nums) - 1, -1, -1):
            num = nums[i]
            res[counts[num - nums_min] - 1] = num
            counts[nums[i] - nums_min] -= 1
        return res

    def sortArray(self, nums: [int]) -> [int]:
        if False:
            for i in range(10):
                print('nop')
        return self.countingSort(nums)
print(Solution().sortArray([3, 0, 4, 2, 5, 1, 3, 1, 4, 5]))