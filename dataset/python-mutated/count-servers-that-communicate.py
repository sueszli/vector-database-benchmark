class Solution(object):

    def countServers(self, grid):
        if False:
            while True:
                i = 10
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        (rows, cols) = ([0] * len(grid), [0] * len(grid[0]))
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                if grid[i][j]:
                    rows[i] += 1
                    cols[j] += 1
        result = 0
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                if grid[i][j] and (rows[i] > 1 or cols[j] > 1):
                    result += 1
        return result