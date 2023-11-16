class Solution(object):

    def removeCoveredIntervals(self, intervals):
        if False:
            print('Hello World!')
        '\n        :type intervals: List[List[int]]\n        :rtype: int\n        '
        intervals.sort(key=lambda x: [x[0], -x[1]])
        (result, max_right) = (0, 0)
        for (left, right) in intervals:
            result += int(right > max_right)
            max_right = max(max_right, right)
        return result