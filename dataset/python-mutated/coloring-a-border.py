import collections

class Solution(object):

    def colorBorder(self, grid, r0, c0, color):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type grid: List[List[int]]\n        :type r0: int\n        :type c0: int\n        :type color: int\n        :rtype: List[List[int]]\n        '
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        (lookup, q, borders) = (set([(r0, c0)]), collections.deque([(r0, c0)]), [])
        while q:
            (r, c) = q.popleft()
            is_border = False
            for direction in directions:
                (nr, nc) = (r + direction[0], c + direction[1])
                if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and (grid[nr][nc] == grid[r][c])):
                    is_border = True
                    continue
                if (nr, nc) in lookup:
                    continue
                lookup.add((nr, nc))
                q.append((nr, nc))
            if is_border:
                borders.append((r, c))
        for (r, c) in borders:
            grid[r][c] = color
        return grid