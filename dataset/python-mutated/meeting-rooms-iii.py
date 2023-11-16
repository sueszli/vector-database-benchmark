import heapq

class Solution(object):

    def mostBooked(self, n, meetings):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type meetings: List[List[int]]\n        :rtype: int\n        '
        meetings.sort()
        min_heap = [(meetings[0][0], i) for i in xrange(n)]
        result = [0] * n
        for (s, e) in meetings:
            while min_heap and min_heap[0][0] < s:
                (_, i) = heapq.heappop(min_heap)
                heapq.heappush(min_heap, (s, i))
            (e2, i) = heapq.heappop(min_heap)
            heapq.heappush(min_heap, (e2 + (e - s), i))
            result[i] += 1
        return max(xrange(n), key=lambda x: result[x])
import heapq

class Solution2(object):

    def mostBooked(self, n, meetings):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type meetings: List[List[int]]\n        :rtype: \n        '
        meetings.sort()
        (unused, used) = (range(n), [])
        result = [0] * n
        for (s, e) in meetings:
            while used and used[0][0] <= s:
                (_, i) = heapq.heappop(used)
                heapq.heappush(unused, i)
            if unused:
                i = heapq.heappop(unused)
                heapq.heappush(used, (e, i))
            else:
                (e2, i) = heapq.heappop(used)
                heapq.heappush(used, (e2 + (e - s), i))
            result[i] += 1
        return max(xrange(n), key=lambda x: result[x])