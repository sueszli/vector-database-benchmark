import heapq

class Solution(object):

    def maxKelements(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def ceil_divide(a, b):
            if False:
                return 10
            return (a + b - 1) // b
        result = 0
        for (i, x) in enumerate(nums):
            nums[i] = -x
        heapq.heapify(nums)
        for _ in xrange(k):
            if not nums:
                break
            x = -heapq.heappop(nums)
            result += x
            nx = ceil_divide(x, 3)
            if not nx:
                continue
            heapq.heappush(nums, -nx)
        return result
import heapq

class Solution2(object):

    def maxKelements(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def ceil_divide(a, b):
            if False:
                i = 10
                return i + 15
            return (a + b - 1) // b
        result = 0
        for (i, x) in enumerate(nums):
            nums[i] = -x
        heapq.heapify(nums)
        for _ in xrange(k):
            x = -heapq.heappop(nums)
            result += x
            heapq.heappush(nums, -ceil_divide(x, 3))
        return result