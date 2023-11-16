class RLEIterator(object):

    def __init__(self, A):
        if False:
            return 10
        '\n        :type A: List[int]\n        '
        self.__A = A
        self.__i = 0
        self.__cnt = 0

    def next(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: int\n        '
        while self.__i < len(self.__A):
            if n > self.__A[self.__i] - self.__cnt:
                n -= self.__A[self.__i] - self.__cnt
                self.__cnt = 0
                self.__i += 2
            else:
                self.__cnt += n
                return self.__A[self.__i + 1]
        return -1