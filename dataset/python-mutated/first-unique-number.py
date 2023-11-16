import collections

class FirstUnique(object):

    def __init__(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        '
        self.__q = collections.OrderedDict()
        self.__dup = set()
        for num in nums:
            self.add(num)

    def showFirstUnique(self):
        if False:
            return 10
        '\n        :rtype: int\n        '
        if self.__q:
            return next(iter(self.__q))
        return -1

    def add(self, value):
        if False:
            while True:
                i = 10
        '\n        :type value: int\n        :rtype: None\n        '
        if value not in self.__dup and value not in self.__q:
            self.__q[value] = None
            return
        if value in self.__q:
            self.__q.pop(value)
            self.__dup.add(value)