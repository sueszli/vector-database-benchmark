class Solution(object):

    def largestLocal(self, grid):
        if False:
            return 10
        '\n        :type grid: List[List[int]]\n        :rtype: List[List[int]]\n        '

        def find_max(i, j):
            if False:
                while True:
                    i = 10
            return max((grid[ni][nj] for ni in xrange(i, i + 3) for nj in xrange(j, j + 3)))
        return [[find_max(i, j) for j in xrange(len(grid[0]) - 2)] for i in xrange(len(grid) - 2)]