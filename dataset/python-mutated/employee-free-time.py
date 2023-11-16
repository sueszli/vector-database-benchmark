import heapq

class Interval(object):

    def __init__(self, s=0, e=0):
        if False:
            print('Hello World!')
        self.start = s
        self.end = e

class Solution(object):

    def employeeFreeTime(self, schedule):
        if False:
            print('Hello World!')
        '\n        :type schedule: List[List[Interval]]\n        :rtype: List[Interval]\n        '
        result = []
        min_heap = [(emp[0].start, eid, 0) for (eid, emp) in enumerate(schedule)]
        heapq.heapify(min_heap)
        last_end = -1
        while min_heap:
            (t, eid, i) = heapq.heappop(min_heap)
            if 0 <= last_end < t:
                result.append(Interval(last_end, t))
            last_end = max(last_end, schedule[eid][i].end)
            if i + 1 < len(schedule[eid]):
                heapq.heappush(min_heap, (schedule[eid][i + 1].start, eid, i + 1))
        return result