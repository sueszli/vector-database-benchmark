class Solution(object):

    def maxAreaOfIsland(self, grid):
        if False:
            while True:
                i = 10
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        directions = [[-1, 0], [1, 0], [0, 1], [0, -1]]

        def dfs(i, j, grid, area):
            if False:
                for i in range(10):
                    print('nop')
            if not (0 <= i < len(grid) and 0 <= j < len(grid[0]) and (grid[i][j] > 0)):
                return False
            grid[i][j] *= -1
            area[0] += 1
            for d in directions:
                dfs(i + d[0], j + d[1], grid, area)
            return True
        result = 0
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                area = [0]
                if dfs(i, j, grid, area):
                    result = max(result, area[0])
        return result