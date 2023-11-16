class Solution(object):

    def minScore(self, grid):
        if False:
            return 10
        '\n        :type grid: List[List[int]]\n        :rtype: List[List[int]]\n        '
        idxs = [(i, j) for i in xrange(len(grid)) for j in xrange(len(grid[0]))]
        idxs.sort(key=lambda x: grid[x[0]][x[1]])
        (row_max, col_max) = ([0] * len(grid), [0] * len(grid[0]))
        for (i, j) in idxs:
            grid[i][j] = row_max[i] = col_max[j] = max(row_max[i], col_max[j]) + 1
        return grid