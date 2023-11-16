class Solution(object):

    def intersectionSizeTwo(self, intervals):
        if False:
            while True:
                i = 10
        '\n        :type intervals: List[List[int]]\n        :rtype: int\n        '
        intervals.sort(key=lambda s_e: (s_e[0], -s_e[1]))
        cnts = [2] * len(intervals)
        result = 0
        while intervals:
            ((start, _), cnt) = (intervals.pop(), cnts.pop())
            for s in xrange(start, start + cnt):
                for i in xrange(len(intervals)):
                    if cnts[i] and s <= intervals[i][1]:
                        cnts[i] -= 1
            result += cnt
        return result