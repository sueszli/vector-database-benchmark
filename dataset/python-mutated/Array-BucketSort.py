class Solution:

    def insertionSort(self, nums: [int]) -> [int]:
        if False:
            return 10
        for i in range(1, len(nums)):
            temp = nums[i]
            j = i
            while j > 0 and nums[j - 1] > temp:
                nums[j] = nums[j - 1]
                j -= 1
            nums[j] = temp
        return nums

    def bucketSort(self, nums: [int], bucket_size=5) -> [int]:
        if False:
            while True:
                i = 10
        (nums_min, nums_max) = (min(nums), max(nums))
        bucket_count = (nums_max - nums_min) // bucket_size + 1
        buckets = [[] for _ in range(bucket_count)]
        for num in nums:
            buckets[(num - nums_min) // bucket_size].append(num)
        res = []
        for bucket in buckets:
            self.insertionSort(bucket)
            res.extend(bucket)
        return res

    def sortArray(self, nums: [int]) -> [int]:
        if False:
            print('Hello World!')
        return self.bucketSort(nums)
print(Solution().sortArray([39, 49, 8, 13, 22, 15, 10, 30, 5, 44]))