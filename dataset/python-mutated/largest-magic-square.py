class Solution(object):

    def largestMagicSquare(self, grid):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '

        def get_sum(prefix, a, b):
            if False:
                print('Hello World!')
            return prefix[b + 1] - prefix[a]

        def check(grid, prefix_row, prefix_col, l, i, j):
            if False:
                return 10
            (diag, anti_diag) = (0, 0)
            for d in xrange(l):
                diag += grid[i + d][j + d]
                anti_diag += grid[i + d][j + l - 1 - d]
            if diag != anti_diag:
                return False
            for ni in xrange(i, i + l):
                if diag != get_sum(prefix_row[ni], j, j + l - 1):
                    return False
            for nj in xrange(j, j + l):
                if diag != get_sum(prefix_col[nj], i, i + l - 1):
                    return False
            return True
        prefix_row = [[0] * (len(grid[0]) + 1) for _ in xrange(len(grid))]
        prefix_col = [[0] * (len(grid) + 1) for _ in xrange(len(grid[0]))]
        for i in xrange(len(grid)):
            for j in xrange(len(grid[0])):
                prefix_row[i][j + 1] = prefix_row[i][j] + grid[i][j]
                prefix_col[j][i + 1] = prefix_col[j][i] + grid[i][j]
        for l in reversed(xrange(1, min(len(grid), len(grid[0])) + 1)):
            for i in xrange(len(grid) - (l - 1)):
                for j in xrange(len(grid[0]) - (l - 1)):
                    if check(grid, prefix_row, prefix_col, l, i, j):
                        return l
        return 1