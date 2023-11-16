import itertools

class Solution(object):

    def maxIncreaseKeepingSkyline(self, grid):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        row_maxes = [max(row) for row in grid]
        col_maxes = [max(col) for col in itertools.izip(*grid)]
        return sum((min(row_maxes[r], col_maxes[c]) - val for (r, row) in enumerate(grid) for (c, val) in enumerate(row)))