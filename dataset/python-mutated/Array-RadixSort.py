class Solution:

    def radixSort(self, nums: [int]) -> [int]:
        if False:
            i = 10
            return i + 15
        size = len(str(max(nums)))
        for i in range(size):
            buckets = [[] for _ in range(10)]
            for num in nums:
                buckets[num // 10 ** i % 10].append(num)
            nums.clear()
            for bucket in buckets:
                for num in bucket:
                    nums.append(num)
        return nums

    def sortArray(self, nums: [int]) -> [int]:
        if False:
            i = 10
            return i + 15
        return self.radixSort(nums)
print(Solution().sortArray([692, 924, 969, 503, 871, 704, 542, 436]))