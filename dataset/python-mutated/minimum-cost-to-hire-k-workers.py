import itertools
import heapq

class Solution(object):

    def mincostToHireWorkers(self, quality, wage, K):
        if False:
            while True:
                i = 10
        '\n        :type quality: List[int]\n        :type wage: List[int]\n        :type K: int\n        :rtype: float\n        '
        (result, qsum) = (float('inf'), 0)
        max_heap = []
        for (r, q) in sorted(([float(w) / q, q] for (w, q) in itertools.izip(wage, quality))):
            qsum += q
            heapq.heappush(max_heap, -q)
            if len(max_heap) > K:
                qsum -= -heapq.heappop(max_heap)
            if len(max_heap) == K:
                result = min(result, qsum * r)
        return result