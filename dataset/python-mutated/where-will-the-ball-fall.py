class Solution(object):

    def findBall(self, grid):
        if False:
            return 10
        '\n        :type grid: List[List[int]]\n        :rtype: List[int]\n        '
        result = []
        for c in xrange(len(grid[0])):
            for r in xrange(len(grid)):
                nc = c + grid[r][c]
                if not (0 <= nc < len(grid[0]) and grid[r][nc] == grid[r][c]):
                    c = -1
                    break
                c = nc
            result.append(c)
        return result