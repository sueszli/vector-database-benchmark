import heapq

class Solution(object):

    def smallestRange(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[List[int]]\n        :rtype: List[int]\n        '
        (left, right) = (float('inf'), float('-inf'))
        min_heap = []
        for row in nums:
            left = min(left, row[0])
            right = max(right, row[0])
            it = iter(row)
            heapq.heappush(min_heap, (next(it, None), it))
        result = (left, right)
        while min_heap:
            (val, it) = heapq.heappop(min_heap)
            val = next(it, None)
            if val is None:
                break
            heapq.heappush(min_heap, (val, it))
            (left, right) = (min_heap[0][0], max(right, val))
            if right - left < result[1] - result[0]:
                result = (left, right)
        return result