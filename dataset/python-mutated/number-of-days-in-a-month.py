class Solution(object):

    def numberOfDays(self, Y, M):
        if False:
            print('Hello World!')
        '\n        :type Y: int\n        :type M: int\n        :rtype: int\n        '
        leap = 1 if Y % 4 == 0 and Y % 100 != 0 or Y % 400 == 0 else 0
        return 28 + leap if M == 2 else 31 - (M - 1) % 7 % 2