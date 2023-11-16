from sortedcontainers import SortedList

class CountIntervals(object):

    def __init__(self):
        if False:
            return 10
        self.__sl = SortedList()
        self.__cnt = 0

    def add(self, left, right):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type left: int\n        :type right: int\n        :rtype: None\n        '
        i = self.__sl.bisect_right((left,))
        if i - 1 >= 0 and self.__sl[i - 1][1] + 1 >= left:
            i -= 1
            left = self.__sl[i][0]
        to_remove = []
        for i in xrange(i, len(self.__sl)):
            if not right + 1 >= self.__sl[i][0]:
                break
            right = max(right, self.__sl[i][1])
            self.__cnt -= self.__sl[i][1] - self.__sl[i][0] + 1
            to_remove.append(i)
        while to_remove:
            del self.__sl[to_remove.pop()]
        self.__sl.add((left, right))
        self.__cnt += right - left + 1

    def count(self):
        if False:
            print('Hello World!')
        '\n        :rtype: int\n        '
        return self.__cnt