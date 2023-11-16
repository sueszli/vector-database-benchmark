class Solution(object):

    def minSubArrayLen(self, s, nums):
        if False:
            print('Hello World!')
        start = 0
        sum = 0
        min_size = float('inf')
        for i in xrange(len(nums)):
            sum += nums[i]
            while sum >= s:
                min_size = min(min_size, i - start + 1)
                sum -= nums[start]
                start += 1
        return min_size if min_size != float('inf') else 0

class Solution2(object):

    def minSubArrayLen(self, s, nums):
        if False:
            for i in range(10):
                print('nop')
        min_size = float('inf')
        sum_from_start = [n for n in nums]
        for i in xrange(len(sum_from_start) - 1):
            sum_from_start[i + 1] += sum_from_start[i]
        for i in xrange(len(sum_from_start)):
            end = self.binarySearch(lambda x, y: x <= y, sum_from_start, i, len(sum_from_start), sum_from_start[i] - nums[i] + s)
            if end < len(sum_from_start):
                min_size = min(min_size, end - i + 1)
        return min_size if min_size != float('inf') else 0

    def binarySearch(self, compare, A, start, end, target):
        if False:
            return 10
        while start < end:
            mid = start + (end - start) / 2
            if compare(target, A[mid]):
                end = mid
            else:
                start = mid + 1
        return start