import heapq

class Solution(object):

    def maxTwoEvents(self, events):
        if False:
            i = 10
            return i + 15
        '\n        :type events: List[List[int]]\n        :rtype: int\n        '
        events.sort()
        result = best = 0
        min_heap = []
        for (left, right, v) in events:
            heapq.heappush(min_heap, (right, v))
            while min_heap and min_heap[0][0] < left:
                best = max(best, heapq.heappop(min_heap)[1])
            result = max(result, best + v)
        return result