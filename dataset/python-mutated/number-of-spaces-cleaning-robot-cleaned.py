class Solution(object):

    def numberOfCleanRooms(self, room):
        if False:
            print('Hello World!')
        '\n        :type room: List[List[int]]\n        :rtype: int\n        '
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        result = r = c = d = 0
        while not room[r][c] & 1 << d + 1:
            result += room[r][c] >> 1 == 0
            room[r][c] |= 1 << d + 1
            (dr, dc) = directions[d]
            (nr, nc) = (r + dr, c + dc)
            if 0 <= nr < len(room) and 0 <= nc < len(room[0]) and (not room[nr][nc] & 1):
                (r, c) = (nr, nc)
            else:
                d = (d + 1) % 4
        return result