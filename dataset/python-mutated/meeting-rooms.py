class Solution(object):

    def canAttendMeetings(self, intervals):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type intervals: List[List[int]]\n        :rtype: bool\n        '
        intervals.sort(key=lambda x: x[0])
        for i in xrange(1, len(intervals)):
            if intervals[i][0] < intervals[i - 1][1]:
                return False
        return True