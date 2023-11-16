import itertools
import heapq

class Solution(object):

    def maxPerformance(self, n, speed, efficiency, k):
        if False:
            return 10
        '\n        :type n: int\n        :type speed: List[int]\n        :type efficiency: List[int]\n        :type k: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        (result, s_sum) = (0, 0)
        min_heap = []
        for (e, s) in sorted(itertools.izip(efficiency, speed), reverse=True):
            s_sum += s
            heapq.heappush(min_heap, s)
            if len(min_heap) > k:
                s_sum -= heapq.heappop(min_heap)
            result = max(result, s_sum * e)
        return result % MOD