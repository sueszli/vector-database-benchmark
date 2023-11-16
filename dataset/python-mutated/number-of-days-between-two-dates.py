class Solution(object):

    def __init__(self):
        if False:
            while True:
                i = 10

        def dayOfMonth(M):
            if False:
                i = 10
                return i + 15
            return 28 if M == 2 else 31 - (M - 1) % 7 % 2
        self.__lookup = [0] * 12
        for M in xrange(1, len(self.__lookup)):
            self.__lookup[M] += self.__lookup[M - 1] + dayOfMonth(M)

    def daysBetweenDates(self, date1, date2):
        if False:
            print('Hello World!')
        '\n        :type date1: str\n        :type date2: str\n        :rtype: int\n        '

        def num_days(date):
            if False:
                while True:
                    i = 10
            (Y, M, D) = map(int, date.split('-'))
            leap = 1 if M > 2 and (Y % 4 == 0 and Y % 100 != 0 or Y % 400 == 0) else 0
            return (Y - 1) * 365 + ((Y - 1) // 4 - (Y - 1) // 100 + (Y - 1) // 400) + self.__lookup[M - 1] + D + leap
        return abs(num_days(date1) - num_days(date2))
import datetime

class Solution2(object):

    def daysBetweenDates(self, date1, date2):
        if False:
            i = 10
            return i + 15
        delta = datetime.datetime.strptime(date1, '%Y-%m-%d')
        delta -= datetime.datetime.strptime(date2, '%Y-%m-%d')
        return abs(delta.days)