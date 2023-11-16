class Solution(object):

    def clumsy(self, N):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type N: int\n        :rtype: int\n        '
        if N <= 2:
            return N
        if N <= 4:
            return N + 3
        if N % 4 == 0:
            return N + 1
        elif N % 4 <= 2:
            return N + 2
        return N - 1