import bisect

class RangeModule(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__intervals = []

    def addRange(self, left, right):
        if False:
            i = 10
            return i + 15
        '\n        :type left: int\n        :type right: int\n        :rtype: void\n        '
        tmp = []
        i = 0
        for interval in self.__intervals:
            if right < interval[0]:
                tmp.append((left, right))
                break
            elif interval[1] < left:
                tmp.append(interval)
            else:
                left = min(left, interval[0])
                right = max(right, interval[1])
            i += 1
        if i == len(self.__intervals):
            tmp.append((left, right))
        while i < len(self.__intervals):
            tmp.append(self.__intervals[i])
            i += 1
        self.__intervals = tmp

    def queryRange(self, left, right):
        if False:
            return 10
        '\n        :type left: int\n        :type right: int\n        :rtype: bool\n        '
        i = bisect.bisect_left(self.__intervals, (left, float('inf')))
        if i:
            i -= 1
        return bool(self.__intervals) and self.__intervals[i][0] <= left and (right <= self.__intervals[i][1])

    def removeRange(self, left, right):
        if False:
            i = 10
            return i + 15
        '\n        :type left: int\n        :type right: int\n        :rtype: void\n        '
        tmp = []
        for interval in self.__intervals:
            if interval[1] <= left or interval[0] >= right:
                tmp.append(interval)
            else:
                if interval[0] < left:
                    tmp.append((interval[0], left))
                if right < interval[1]:
                    tmp.append((right, interval[1]))
        self.__intervals = tmp