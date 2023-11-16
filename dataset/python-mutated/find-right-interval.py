import bisect

class Solution(object):

    def findRightInterval(self, intervals):
        if False:
            return 10
        '\n        :type intervals: List[Interval]\n        :rtype: List[int]\n        '
        sorted_intervals = sorted(((interval.start, i) for (i, interval) in enumerate(intervals)))
        result = []
        for interval in intervals:
            idx = bisect.bisect_left(sorted_intervals, (interval.end,))
            result.append(sorted_intervals[idx][1] if idx < len(sorted_intervals) else -1)
        return result