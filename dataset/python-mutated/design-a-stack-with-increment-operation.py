class CustomStack(object):

    def __init__(self, maxSize):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type maxSize: int\n        '
        self.__max_size = maxSize
        self.__stk = []

    def push(self, x):
        if False:
            while True:
                i = 10
        '\n        :type x: int\n        :rtype: None\n        '
        if len(self.__stk) == self.__max_size:
            return
        self.__stk.append([x, 0])

    def pop(self):
        if False:
            while True:
                i = 10
        '\n        :rtype: int\n        '
        if not self.__stk:
            return -1
        (x, inc) = self.__stk.pop()
        if self.__stk:
            self.__stk[-1][1] += inc
        return x + inc

    def increment(self, k, val):
        if False:
            return 10
        '\n        :type k: int\n        :type val: int\n        :rtype: None\n        '
        i = min(len(self.__stk), k) - 1
        if i >= 0:
            self.__stk[i][1] += val