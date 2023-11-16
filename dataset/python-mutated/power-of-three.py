import math

class Solution(object):

    def __init__(self):
        if False:
            return 10
        self.__max_log3 = int(math.log(2147483647) / math.log(3))
        self.__max_pow3 = 3 ** self.__max_log3

    def isPowerOfThree(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: bool\n        '
        return n > 0 and self.__max_pow3 % n == 0

class Solution2(object):

    def isPowerOfThree(self, n):
        if False:
            print('Hello World!')
        return n > 0 and (math.log10(n) / math.log10(3)).is_integer()