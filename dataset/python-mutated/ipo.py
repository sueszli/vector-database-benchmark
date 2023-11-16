import heapq

class Solution(object):

    def findMaximizedCapital(self, k, W, Profits, Capital):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type k: int\n        :type W: int\n        :type Profits: List[int]\n        :type Capital: List[int]\n        :rtype: int\n        '
        curr = []
        future = sorted(zip(Capital, Profits), reverse=True)
        for _ in xrange(k):
            while future and future[-1][0] <= W:
                heapq.heappush(curr, -future.pop()[1])
            if curr:
                W -= heapq.heappop(curr)
        return W