class Solution(object):

    def getMaximumGold(self, grid):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def backtracking(grid, i, j):
            if False:
                while True:
                    i = 10
            result = 0
            grid[i][j] *= -1
            for (dx, dy) in directions:
                (ni, nj) = (i + dx, j + dy)
                if not (0 <= ni < len(grid) and 0 <= nj < len(grid[0]) and (grid[ni][nj] > 0)):
                    continue
                result = max(result, backtracking(grid, ni, nj))
            grid[i][j] *= -1
            return grid[i][j] + result
        result = 0
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                if grid[i][j]:
                    result = max(result, backtracking(grid, i, j))
        return result