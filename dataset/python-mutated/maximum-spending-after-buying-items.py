import heapq

class Solution(object):

    def maxSpending(self, values):
        if False:
            i = 10
            return i + 15
        '\n        :type values: List[List[int]]\n        :rtype: int\n        '
        (m, n) = (len(values), len(values[0]))
        min_heap = [(values[i].pop(), i) for i in xrange(m)]
        heapq.heapify(min_heap)
        result = 0
        for d in xrange(1, m * n + 1):
            (x, i) = heapq.heappop(min_heap)
            result += x * d
            if values[i]:
                heapq.heappush(min_heap, (values[i].pop(), i))
        return result