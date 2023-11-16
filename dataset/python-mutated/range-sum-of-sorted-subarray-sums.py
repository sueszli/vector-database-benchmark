class Solution(object):

    def rangeSum(self, nums, n, left, right):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type n: int\n        :type left: int\n        :type right: int\n        :rtype: int\n        '

        def countUntil(nums, target):
            if False:
                for i in range(10):
                    print('nop')
            (result, curr, left) = (0, 0, 0)
            for right in xrange(len(nums)):
                curr += nums[right]
                while curr > target:
                    curr -= nums[left]
                    left += 1
                result += right - left + 1
            return result

        def sumUntil(nums, prefix, target):
            if False:
                return 10
            (result, curr, total, left) = (0, 0, 0, 0)
            for right in xrange(len(nums)):
                curr += nums[right]
                total += nums[right] * (right - left + 1)
                while curr > target:
                    curr -= nums[left]
                    total -= prefix[right + 1] - prefix[left - 1 + 1]
                    left += 1
                result += total
            return result

        def sumLessOrEqualTo(prefix, nums, left, right, count):
            if False:
                while True:
                    i = 10
            while left <= right:
                mid = left + (right - left) // 2
                if countUntil(nums, mid) - count >= 0:
                    right = mid - 1
                else:
                    left = mid + 1
            return sumUntil(nums, prefix, left) - left * (countUntil(nums, left) - count)
        MOD = 10 ** 9 + 7
        prefix = [0] * (len(nums) + 1)
        for i in xrange(len(nums)):
            prefix[i + 1] = prefix[i] + nums[i]
        (m, M) = (min(nums), sum(nums))
        return (sumLessOrEqualTo(prefix, nums, m, M, right) - sumLessOrEqualTo(prefix, nums, m, M, left - 1)) % MOD
import heapq

class Solution2(object):

    def rangeSum(self, nums, n, left, right):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type n: int\n        :type left: int\n        :type right: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        min_heap = []
        for (i, num) in enumerate(nums, 1):
            heapq.heappush(min_heap, (num, i))
        result = 0
        for i in xrange(1, right + 1):
            (total, j) = heapq.heappop(min_heap)
            if i >= left:
                result = (result + total) % MOD
            if j + 1 <= n:
                heapq.heappush(min_heap, (total + nums[j], j + 1))
        return result