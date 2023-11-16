from random import randint
from collections import defaultdict

class RandomizedCollection(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        Initialize your data structure here.\n        '
        self.__list = []
        self.__used = defaultdict(list)

    def insert(self, val):
        if False:
            print('Hello World!')
        '\n        Inserts a value to the collection. Returns true if the collection did not already contain the specified element.\n        :type val: int\n        :rtype: bool\n        '
        has = val in self.__used
        self.__list += ((val, len(self.__used[val])),)
        self.__used[val] += (len(self.__list) - 1,)
        return not has

    def remove(self, val):
        if False:
            return 10
        '\n        Removes a value from the collection. Returns true if the collection contained the specified element.\n        :type val: int\n        :rtype: bool\n        '
        if val not in self.__used:
            return False
        self.__used[self.__list[-1][0]][self.__list[-1][1]] = self.__used[val][-1]
        (self.__list[self.__used[val][-1]], self.__list[-1]) = (self.__list[-1], self.__list[self.__used[val][-1]])
        self.__used[val].pop()
        if not self.__used[val]:
            self.__used.pop(val)
        self.__list.pop()
        return True

    def getRandom(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a random element from the collection.\n        :rtype: int\n        '
        return self.__list[randint(0, len(self.__list) - 1)][0]