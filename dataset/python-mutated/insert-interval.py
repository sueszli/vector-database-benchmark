class Solution(object):

    def insert(self, intervals, newInterval):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type intervals: List[List[int]]\n        :type newInterval: List[int]\n        :rtype: List[List[int]]\n        '
        result = []
        i = 0
        while i < len(intervals) and newInterval[0] > intervals[i][1]:
            result += (intervals[i],)
            i += 1
        while i < len(intervals) and newInterval[1] >= intervals[i][0]:
            newInterval = [min(newInterval[0], intervals[i][0]), max(newInterval[1], intervals[i][1])]
            i += 1
        result.append(newInterval)
        result.extend(intervals[i:])
        return result