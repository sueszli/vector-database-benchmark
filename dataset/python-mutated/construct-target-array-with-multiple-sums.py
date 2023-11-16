import heapq

class Solution(object):

    def isPossible(self, target):
        if False:
            i = 10
            return i + 15
        '\n        :type target: List[int]\n        :rtype: bool\n        '
        total = sum(target)
        max_heap = [-x for x in target]
        heapq.heapify(max_heap)
        while total != len(target):
            y = -heapq.heappop(max_heap)
            remain = total - y
            x = y - remain
            if x <= 0:
                return False
            if x > remain:
                x = x % remain + remain
            heapq.heappush(max_heap, -x)
            total = x + remain
        return True