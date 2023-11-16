class Solution(object):

    def minimumOneBitOperations(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: int\n        '

        def gray_to_binary(n):
            if False:
                print('Hello World!')
            result = 0
            while n:
                result ^= n
                n >>= 1
            return result
        return gray_to_binary(n)

class Solution2(object):

    def minimumOneBitOperations(self, n):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :rtype: int\n        '
        result = 0
        while n:
            result = -result - (n ^ n - 1)
            n &= n - 1
        return abs(result)