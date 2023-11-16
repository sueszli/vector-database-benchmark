class Solution(object):

    def removeOnes(self, grid):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type grid: List[List[int]]\n        :rtype: bool\n        '
        return all((grid[i] == grid[0] or all((grid[i][j] != grid[0][j] for j in xrange(len(grid[0])))) for i in xrange(1, len(grid))))