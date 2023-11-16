class Solution(object):

    def eraseOverlapIntervals(self, intervals):
        if False:
            while True:
                i = 10
        '\n        :type intervals: List[Interval]\n        :rtype: int\n        '
        intervals.sort(key=lambda interval: interval.start)
        (result, prev) = (0, 0)
        for i in xrange(1, len(intervals)):
            if intervals[i].start < intervals[prev].end:
                if intervals[i].end < intervals[prev].end:
                    prev = i
                result += 1
            else:
                prev = i
        return result