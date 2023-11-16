class Solution(object):

    def countNegatives(self, grid):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        (result, c) = (0, len(grid[0]) - 1)
        for row in grid:
            while c >= 0 and row[c] < 0:
                c -= 1
            result += len(grid[0]) - 1 - c
        return result