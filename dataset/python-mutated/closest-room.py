from sortedcontainers import SortedList

class Solution(object):

    def closestRoom(self, rooms, queries):
        if False:
            i = 10
            return i + 15
        '\n        :type rooms: List[List[int]]\n        :type queries: List[List[int]]\n        :rtype: List[int]\n        '

        def find_closest(ids, r):
            if False:
                while True:
                    i = 10
            (result, min_dist) = (-1, float('inf'))
            i = ids.bisect_right(r)
            if i - 1 >= 0 and abs(ids[i - 1] - r) < min_dist:
                min_dist = abs(ids[i - 1] - r)
                result = ids[i - 1]
            if i < len(ids) and abs(ids[i] - r) < min_dist:
                min_dist = abs(ids[i] - r)
                result = ids[i]
            return result
        rooms.sort(key=lambda x: x[1], reverse=True)
        for (i, q) in enumerate(queries):
            q.append(i)
        queries.sort(key=lambda x: x[1], reverse=True)
        ids = SortedList()
        i = 0
        result = [-1] * len(queries)
        for (r, s, idx) in queries:
            while i < len(rooms) and rooms[i][1] >= s:
                ids.add(rooms[i][0])
                i += 1
            result[idx] = find_closest(ids, r)
        return result
from sortedcontainers import SortedList

class Solution2(object):

    def closestRoom(self, rooms, queries):
        if False:
            while True:
                i = 10
        '\n        :type rooms: List[List[int]]\n        :type queries: List[List[int]]\n        :rtype: List[int]\n        '

        def find_closest(ids, r):
            if False:
                print('Hello World!')
            (result, min_dist) = (-1, float('inf'))
            i = ids.bisect_right(r)
            if i - 1 >= 0 and abs(ids[i - 1] - r) < min_dist:
                min_dist = abs(ids[i - 1] - r)
                result = ids[i - 1]
            if i < len(ids) and abs(ids[i] - r) < min_dist:
                min_dist = abs(ids[i] - r)
                result = ids[i]
            return result
        rooms.sort(key=lambda x: x[1])
        for (i, q) in enumerate(queries):
            q.append(i)
        queries.sort(key=lambda x: x[1])
        ids = SortedList((i for (i, _) in rooms))
        i = 0
        result = [-1] * len(queries)
        for (r, s, idx) in queries:
            while i < len(rooms) and rooms[i][1] < s:
                ids.remove(rooms[i][0])
                i += 1
            result[idx] = find_closest(ids, r)
        return result