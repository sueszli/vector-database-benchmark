import collections
import heapq

class Solution(object):

    def scheduleCourse(self, courses):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type courses: List[List[int]]\n        :rtype: int\n        '
        courses.sort(key=lambda t_end: t_end[1])
        max_heap = []
        now = 0
        for (t, end) in courses:
            now += t
            heapq.heappush(max_heap, -t)
            if now > end:
                now += heapq.heappop(max_heap)
        return len(max_heap)