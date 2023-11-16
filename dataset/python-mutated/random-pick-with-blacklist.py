import random

class Solution(object):

    def __init__(self, N, blacklist):
        if False:
            while True:
                i = 10
        '\n        :type N: int\n        :type blacklist: List[int]\n        '
        self.__n = N - len(blacklist)
        self.__lookup = {}
        white = iter(set(range(self.__n, N)) - set(blacklist))
        for black in blacklist:
            if black < self.__n:
                self.__lookup[black] = next(white)

    def pick(self):
        if False:
            return 10
        '\n        :rtype: int\n        '
        index = random.randint(0, self.__n - 1)
        return self.__lookup[index] if index in self.__lookup else index
import random

class Solution2(object):

    def __init__(self, N, blacklist):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type N: int\n        :type blacklist: List[int]\n        '
        self.__n = N - len(blacklist)
        blacklist.sort()
        self.__blacklist = blacklist

    def pick(self):
        if False:
            print('Hello World!')
        '\n        :rtype: int\n        '
        index = random.randint(0, self.__n - 1)
        (left, right) = (0, len(self.__blacklist) - 1)
        while left <= right:
            mid = left + (right - left) // 2
            if index + mid < self.__blacklist[mid]:
                right = mid - 1
            else:
                left = mid + 1
        return index + left