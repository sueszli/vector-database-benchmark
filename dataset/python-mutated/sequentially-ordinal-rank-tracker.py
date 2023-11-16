from sortedcontainers import SortedList

class SORTracker(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.__sl = SortedList()
        self.__i = 0

    def add(self, name, score):
        if False:
            return 10
        '\n        :type name: str\n        :type score: int\n        :rtype: None\n        '
        self.__sl.add((-score, name))

    def get(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: str\n        '
        self.__i += 1
        return self.__sl[self.__i - 1][1]