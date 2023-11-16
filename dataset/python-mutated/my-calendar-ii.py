class MyCalendarTwo(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.__overlaps = []
        self.__calendar = []

    def book(self, start, end):
        if False:
            print('Hello World!')
        '\n        :type start: int\n        :type end: int\n        :rtype: bool\n        '
        for (i, j) in self.__overlaps:
            if start < j and end > i:
                return False
        for (i, j) in self.__calendar:
            if start < j and end > i:
                self.__overlaps.append((max(start, i), min(end, j)))
        self.__calendar.append((start, end))
        return True