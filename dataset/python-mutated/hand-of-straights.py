from collections import Counter
from heapq import heapify, heappop

class Solution(object):

    def isNStraightHand(self, hand, W):
        if False:
            i = 10
            return i + 15
        '\n        :type hand: List[int]\n        :type W: int\n        :rtype: bool\n        '
        if len(hand) % W:
            return False
        counts = Counter(hand)
        min_heap = list(hand)
        heapify(min_heap)
        for _ in xrange(len(min_heap) // W):
            while counts[min_heap[0]] == 0:
                heappop(min_heap)
            start = heappop(min_heap)
            for _ in xrange(W):
                counts[start] -= 1
                if counts[start] < 0:
                    return False
                start += 1
        return True