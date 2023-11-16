import itertools

class Solution(object):

    def maxWidthOfVerticalArea(self, points):
        if False:
            i = 10
            return i + 15
        '\n        :type points: List[List[int]]\n        :rtype: int\n        '
        sorted_x = sorted({x for (x, y) in points})
        return max([b - a for (a, b) in itertools.izip(sorted_x, sorted_x[1:])] + [0])