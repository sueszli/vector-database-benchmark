import heapq

class Solution(object):

    def lastStoneWeight(self, stones):
        if False:
            return 10
        '\n        :type stones: List[int]\n        :rtype: int\n        '
        max_heap = [-x for x in stones]
        heapq.heapify(max_heap)
        for i in xrange(len(stones) - 1):
            (x, y) = (-heapq.heappop(max_heap), -heapq.heappop(max_heap))
            heapq.heappush(max_heap, -abs(x - y))
        return -max_heap[0]