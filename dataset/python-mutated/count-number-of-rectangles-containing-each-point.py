import bisect

class Solution(object):

    def countRectangles(self, rectangles, points):
        if False:
            return 10
        '\n        :type rectangles: List[List[int]]\n        :type points: List[List[int]]\n        :rtype: List[int]\n        '
        max_y = max((y for (_, y) in rectangles))
        buckets = [[] for _ in xrange(max_y + 1)]
        for (x, y) in rectangles:
            buckets[y].append(x)
        for bucket in buckets:
            bucket.sort()
        return [sum((len(buckets[y]) - bisect.bisect_left(buckets[y], x) for y in xrange(y, max_y + 1))) for (x, y) in points]