class Bitset(object):

    def __init__(self, size):
        if False:
            while True:
                i = 10
        '\n        :type size: int\n        '
        self.__lookup = [False] * size
        self.__flip = False
        self.__cnt = 0

    def fix(self, idx):
        if False:
            return 10
        '\n        :type idx: int\n        :rtype: None\n        '
        if self.__lookup[idx] == self.__flip:
            self.__lookup[idx] = not self.__lookup[idx]
            self.__cnt += 1

    def unfix(self, idx):
        if False:
            print('Hello World!')
        '\n        :type idx: int\n        :rtype: None\n        '
        if self.__lookup[idx] != self.__flip:
            self.__lookup[idx] = not self.__lookup[idx]
            self.__cnt -= 1

    def flip(self):
        if False:
            return 10
        '\n        :rtype: None\n        '
        self.__flip = not self.__flip
        self.__cnt = len(self.__lookup) - self.__cnt

    def all(self):
        if False:
            print('Hello World!')
        '\n        :rtype: bool\n        '
        return self.__cnt == len(self.__lookup)

    def one(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :rtype: bool\n        '
        return self.__cnt >= 1

    def count(self):
        if False:
            print('Hello World!')
        '\n        :rtype: int\n        '
        return self.__cnt

    def toString(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: str\n        '
        result = [''] * len(self.__lookup)
        for (i, x) in enumerate(self.__lookup):
            result[i] = '1' if x != self.__flip else '0'
        return ''.join(result)