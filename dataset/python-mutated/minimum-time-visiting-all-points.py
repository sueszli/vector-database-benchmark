class Solution(object):

    def minTimeToVisitAllPoints(self, points):
        if False:
            i = 10
            return i + 15
        '\n        :type points: List[List[int]]\n        :rtype: int\n        '
        return sum((max(abs(points[i + 1][0] - points[i][0]), abs(points[i + 1][1] - points[i][1])) for i in xrange(len(points) - 1)))