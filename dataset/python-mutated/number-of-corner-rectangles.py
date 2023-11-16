class Solution(object):

    def countCornerRectangles(self, grid):
        if False:
            print('Hello World!')
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        rows = [[c for (c, val) in enumerate(row) if val] for row in grid]
        result = 0
        for i in xrange(len(rows)):
            lookup = set(rows[i])
            for j in xrange(i):
                count = sum((1 for c in rows[j] if c in lookup))
                result += count * (count - 1) / 2
        return result