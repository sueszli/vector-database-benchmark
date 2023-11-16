class Solution(object):

    def onesMinusZeros(self, grid):
        if False:
            i = 10
            return i + 15
        '\n        :type grid: List[List[int]]\n        :rtype: List[List[int]]\n        '
        rows = [sum((grid[i][j] for j in xrange(len(grid[0])))) for i in xrange(len(grid))]
        cols = [sum((grid[i][j] for i in xrange(len(grid)))) for j in xrange(len(grid[0]))]
        return [[rows[i] + cols[j] - (len(grid) - rows[i]) - (len(grid[0]) - cols[j]) for j in xrange(len(grid[0]))] for i in xrange(len(grid))]