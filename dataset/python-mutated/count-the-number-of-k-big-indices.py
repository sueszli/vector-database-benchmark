import heapq

class Solution(object):

    def kBigIndices(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        right = [False] * len(nums)
        max_heap1 = []
        for i in reversed(xrange(len(nums))):
            if len(max_heap1) == k and nums[i] > -max_heap1[0]:
                right[i] = True
            heapq.heappush(max_heap1, -nums[i])
            if len(max_heap1) == k + 1:
                heapq.heappop(max_heap1)
        result = 0
        max_heap2 = []
        for i in xrange(len(nums)):
            if len(max_heap2) == k and nums[i] > -max_heap2[0] and right[i]:
                result += 1
            heapq.heappush(max_heap2, -nums[i])
            if len(max_heap2) == k + 1:
                heapq.heappop(max_heap2)
        return result
from sortedcontainers import SortedList

class Solution2(object):

    def kBigIndices(self, nums, k):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        (sl1, sl2) = (SortedList(), SortedList(nums))
        result = 0
        for x in nums:
            sl2.remove(x)
            if sl1.bisect_left(x) >= k and sl2.bisect_left(x) >= k:
                result += 1
            sl1.add(x)
        return result