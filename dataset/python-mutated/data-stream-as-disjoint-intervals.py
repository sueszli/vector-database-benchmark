class Interval(object):

    def __init__(self, s=0, e=0):
        if False:
            for i in range(10):
                print('nop')
        self.start = s
        self.end = e

class SummaryRanges(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        '\n        Initialize your data structure here.\n        '
        self.__intervals = []

    def addNum(self, val):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type val: int\n        :rtype: void\n        '

        def upper_bound(nums, target):
            if False:
                for i in range(10):
                    print('nop')
            (left, right) = (0, len(nums) - 1)
            while left <= right:
                mid = left + (right - left) / 2
                if nums[mid].start > target:
                    right = mid - 1
                else:
                    left = mid + 1
            return left
        i = upper_bound(self.__intervals, val)
        (start, end) = (val, val)
        if i != 0 and self.__intervals[i - 1].end + 1 >= val:
            i -= 1
        while i != len(self.__intervals) and end + 1 >= self.__intervals[i].start:
            start = min(start, self.__intervals[i].start)
            end = max(end, self.__intervals[i].end)
            del self.__intervals[i]
        self.__intervals.insert(i, Interval(start, end))

    def getIntervals(self):
        if False:
            return 10
        '\n        :rtype: List[Interval]\n        '
        return self.__intervals