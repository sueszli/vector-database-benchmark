import collections

class Solution(object):

    def leastBricks(self, wall):
        if False:
            while True:
                i = 10
        '\n        :type wall: List[List[int]]\n        :rtype: int\n        '
        widths = collections.defaultdict(int)
        result = len(wall)
        for row in wall:
            width = 0
            for i in xrange(len(row) - 1):
                width += row[i]
                widths[width] += 1
                result = min(result, len(wall) - widths[width])
        return result