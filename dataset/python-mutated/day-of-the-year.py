class Solution(object):

    def __init__(self):
        if False:
            print('Hello World!')

        def dayOfMonth(M):
            if False:
                while True:
                    i = 10
            return 28 if M == 2 else 31 - (M - 1) % 7 % 2
        self.__lookup = [0] * 12
        for M in xrange(1, len(self.__lookup)):
            self.__lookup[M] += self.__lookup[M - 1] + dayOfMonth(M)

    def dayOfYear(self, date):
        if False:
            while True:
                i = 10
        '\n        :type date: str\n        :rtype: int\n        '
        (Y, M, D) = map(int, date.split('-'))
        leap = 1 if M > 2 and (Y % 4 == 0 and Y % 100 != 0 or Y % 400 == 0) else 0
        return self.__lookup[M - 1] + D + leap

class Solution2(object):

    def dayOfYear(self, date):
        if False:
            return 10
        '\n        :type date: str\n        :rtype: int\n        '

        def numberOfDays(Y, M):
            if False:
                print('Hello World!')
            leap = 1 if Y % 4 == 0 and Y % 100 != 0 or Y % 400 == 0 else 0
            return 28 + leap if M == 2 else 31 - (M - 1) % 7 % 2
        (Y, M, result) = map(int, date.split('-'))
        for i in xrange(1, M):
            result += numberOfDays(Y, i)
        return result