class Solution(object):

    def deleteGreatestValue(self, grid):
        if False:
            i = 10
            return i + 15
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        for row in grid:
            row.sort()
        return sum((max((grid[i][j] for i in xrange(len(grid)))) for j in xrange(len(grid[0]))))