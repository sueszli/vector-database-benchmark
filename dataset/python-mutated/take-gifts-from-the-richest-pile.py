import heapq

class Solution(object):

    def pickGifts(self, gifts, k):
        if False:
            print('Hello World!')
        '\n        :type gifts: List[int]\n        :type k: int\n        :rtype: int\n        '
        for (i, x) in enumerate(gifts):
            gifts[i] = -x
        heapq.heapify(gifts)
        for _ in xrange(k):
            x = heapq.heappop(gifts)
            heapq.heappush(gifts, -int((-x) ** 0.5))
        return -sum(gifts)