import heapq

class Solution(object):

    def minBuildTime(self, blocks, split):
        if False:
            while True:
                i = 10
        '\n        :type blocks: List[int]\n        :type split: int\n        :rtype: int\n        '
        heapq.heapify(blocks)
        while len(blocks) != 1:
            (x, y) = (heapq.heappop(blocks), heapq.heappop(blocks))
            heapq.heappush(blocks, y + split)
        return heapq.heappop(blocks)