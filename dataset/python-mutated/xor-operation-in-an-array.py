class Solution(object):

    def xorOperation(self, n, start):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type start: int\n        :rtype: int\n        '

        def xorNums(n, start):
            if False:
                i = 10
                return i + 15

            def xorNumsBeginEven(n, start):
                if False:
                    return 10
                assert start % 2 == 0
                return n // 2 % 2 ^ (start + n - 1 if n % 2 else 0)
            return start ^ xorNumsBeginEven(n - 1, start + 1) if start % 2 else xorNumsBeginEven(n, start)
        return int(n % 2 and start % 2) + 2 * xorNums(n, start // 2)
import operator

class Solution2(object):

    def xorOperation(self, n, start):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type start: int\n        :rtype: int\n        '
        return reduce(operator.xor, (i for i in xrange(start, start + 2 * n, 2)))