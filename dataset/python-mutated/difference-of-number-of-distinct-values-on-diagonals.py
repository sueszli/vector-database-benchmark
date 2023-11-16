class Solution(object):

    def differenceOfDistinctValues(self, grid):
        if False:
            i = 10
            return i + 15
        '\n        :type grid: List[List[int]]\n        :rtype: List[List[int]]\n        '

        def update(i, j):
            if False:
                return 10
            lookup = set()
            for k in xrange(min(len(grid) - i, len(grid[0]) - j)):
                result[i + k][j + k] = len(lookup)
                lookup.add(grid[i + k][j + k])
            lookup.clear()
            for k in reversed(xrange(min(len(grid) - i, len(grid[0]) - j))):
                result[i + k][j + k] = abs(result[i + k][j + k] - len(lookup))
                lookup.add(grid[i + k][j + k])
        result = [[0] * len(grid[0]) for _ in xrange(len(grid))]
        for j in xrange(len(grid[0])):
            update(0, j)
        for i in xrange(1, len(grid)):
            update(i, 0)
        return result