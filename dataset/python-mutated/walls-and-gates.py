from collections import deque

class Solution(object):

    def wallsAndGates(self, rooms):
        if False:
            i = 10
            return i + 15
        '\n        :type rooms: List[List[int]]\n        :rtype: void Do not return anything, modify rooms in-place instead.\n        '
        INF = 2147483647
        q = deque([(i, j) for (i, row) in enumerate(rooms) for (j, r) in enumerate(row) if not r])
        while q:
            (i, j) = q.popleft()
            for (I, J) in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= I < len(rooms) and 0 <= J < len(rooms[0]) and (rooms[I][J] == INF):
                    rooms[I][J] = rooms[i][j] + 1
                    q.append((I, J))