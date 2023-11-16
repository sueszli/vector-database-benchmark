class Solution(object):

    def minMeetingRooms(self, intervals):
        if False:
            while True:
                i = 10
        (result, curr) = (0, 0)
        line = [x for (i, j) in intervals for x in [[i, 1], [j, -1]]]
        line.sort()
        for (_, num) in line:
            curr += num
            result = max(result, curr)
        return result

class Solution2(object):

    def minMeetingRooms(self, intervals):
        if False:
            for i in range(10):
                print('nop')
        (starts, ends) = ([], [])
        for (start, end) in intervals:
            starts.append(start)
            ends.append(end)
        starts.sort()
        ends.sort()
        (s, e) = (0, 0)
        (min_rooms, cnt_rooms) = (0, 0)
        while s < len(starts):
            if starts[s] < ends[e]:
                cnt_rooms += 1
                min_rooms = max(min_rooms, cnt_rooms)
                s += 1
            else:
                cnt_rooms -= 1
                e += 1
        return min_rooms
from heapq import heappush, heappop

class Solution3(object):

    def minMeetingRooms(self, intervals):
        if False:
            return 10
        '\n        :type intervals: List[Interval]\n        :rtype: int\n        '
        if not intervals:
            return 0
        intervals.sort(key=lambda x: x[0])
        free_rooms = []
        heappush(free_rooms, intervals[0][1])
        for interval in intervals[1:]:
            if free_rooms[0] <= interval[0]:
                heappop(free_rooms)
            heappush(free_rooms, interval[1])
        return len(free_rooms)