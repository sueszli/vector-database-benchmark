from random import randint

class RandomizedSet(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize your data structure here.\n        '
        self.__set = []
        self.__used = {}

    def insert(self, val):
        if False:
            while True:
                i = 10
        '\n        Inserts a value to the set. Returns true if the set did not already contain the specified element.\n        :type val: int\n        :rtype: bool\n        '
        if val in self.__used:
            return False
        self.__set += (val,)
        self.__used[val] = len(self.__set) - 1
        return True

    def remove(self, val):
        if False:
            print('Hello World!')
        '\n        Removes a value from the set. Returns true if the set contained the specified element.\n        :type val: int\n        :rtype: bool\n        '
        if val not in self.__used:
            return False
        self.__used[self.__set[-1]] = self.__used[val]
        (self.__set[self.__used[val]], self.__set[-1]) = (self.__set[-1], self.__set[self.__used[val]])
        self.__used.pop(val)
        self.__set.pop()
        return True

    def getRandom(self):
        if False:
            print('Hello World!')
        '\n        Get a random element from the set.\n        :rtype: int\n        '
        return self.__set[randint(0, len(self.__set) - 1)]