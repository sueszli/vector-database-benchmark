import random

class Solution(object):

    def __init__(self, n_rows, n_cols):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n_rows: int\n        :type n_cols: int\n        '
        self.__n_rows = n_rows
        self.__n_cols = n_cols
        self.__n = n_rows * n_cols
        self.__lookup = {}

    def flip(self):
        if False:
            print('Hello World!')
        '\n        :rtype: List[int]\n        '
        self.__n -= 1
        target = random.randint(0, self.__n)
        x = self.__lookup.get(target, target)
        self.__lookup[target] = self.__lookup.get(self.__n, self.__n)
        return divmod(x, self.__n_cols)

    def reset(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: void\n        '
        self.__n = self.__n_rows * self.__n_cols
        self.__lookup = {}