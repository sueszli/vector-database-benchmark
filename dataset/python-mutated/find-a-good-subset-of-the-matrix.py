class Solution(object):

    def goodSubsetofBinaryMatrix(self, grid):
        if False:
            return 10
        '\n        :type grid: List[List[int]]\n        :rtype: List[int]\n        '
        lookup = {}
        for i in xrange(len(grid)):
            mask = reduce(lambda mask, j: mask | grid[i][j] << j, xrange(len(grid[0])), 0)
            if not mask:
                return [i]
            for (mask2, j) in lookup.iteritems():
                if mask2 & mask == 0:
                    return [j, i]
            lookup[mask] = i
        return []