class Solution(object):

    def projectionArea(self, grid):
        if False:
            while True:
                i = 10
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        result = 0
        for i in xrange(len(grid)):
            (max_row, max_col) = (0, 0)
            for j in xrange(len(grid)):
                if grid[i][j]:
                    result += 1
                max_row = max(max_row, grid[i][j])
                max_col = max(max_col, grid[j][i])
            result += max_row + max_col
        return result