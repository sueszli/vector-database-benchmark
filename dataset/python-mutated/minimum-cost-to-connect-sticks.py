import heapq

class Solution(object):

    def connectSticks(self, sticks):
        if False:
            return 10
        '\n        :type sticks: List[int]\n        :rtype: int\n        '
        heapq.heapify(sticks)
        result = 0
        while len(sticks) > 1:
            (x, y) = (heapq.heappop(sticks), heapq.heappop(sticks))
            result += x + y
            heapq.heappush(sticks, x + y)
        return result