import heapq

class Solution(object):

    def minStoneSum(self, piles, k):
        if False:
            while True:
                i = 10
        '\n        :type piles: List[int]\n        :type k: int\n        :rtype: int\n        '
        for (i, x) in enumerate(piles):
            piles[i] = -x
        heapq.heapify(piles)
        for i in xrange(k):
            heapq.heappush(piles, heapq.heappop(piles) // 2)
        return -sum(piles)