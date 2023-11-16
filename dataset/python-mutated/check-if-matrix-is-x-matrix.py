class Solution(object):

    def checkXMatrix(self, grid):
        if False:
            i = 10
            return i + 15
        '\n        :type grid: List[List[int]]\n        :rtype: bool\n        '
        return all(((i - j == 0 or i + j == len(grid) - 1) == (grid[i][j] != 0) for i in xrange(len(grid)) for j in xrange(len(grid[0]))))