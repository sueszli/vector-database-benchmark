class Solution(object):

    def checkValidGrid(self, grid):
        if False:
            return 10
        '\n        :type grid: List[List[int]]\n        :rtype: bool\n        '
        if grid[0][0]:
            return False
        lookup = [None] * (len(grid) * len(grid[0]))
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                lookup[grid[i][j]] = (i, j)
        return all((sorted([abs(lookup[i + 1][0] - lookup[i][0]), abs(lookup[i + 1][1] - lookup[i][1])]) == [1, 2] for i in xrange(len(lookup) - 1)))

class Solution2(object):

    def checkValidGrid(self, grid):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type grid: List[List[int]]\n        :rtype: bool\n        '
        lookup = {grid[i][j]: (i, j) for i in xrange(len(grid)) for j in xrange(len(grid[0]))}
        return grid[0][0] == 0 and all((sorted([abs(lookup[i + 1][0] - lookup[i][0]), abs(lookup[i + 1][1] - lookup[i][1])]) == [1, 2] for i in xrange(len(lookup) - 1)))