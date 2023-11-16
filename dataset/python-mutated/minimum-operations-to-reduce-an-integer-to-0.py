class Solution(object):

    def minOperations(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: int\n        '

        def popcount(x):
            if False:
                print('Hello World!')
            return bin(x)[2:].count('1')
        return popcount(n ^ n * 3)

class Solution2(object):

    def minOperations(self, n):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :rtype: int\n        '
        result = 0
        while n:
            if n & 1:
                n >>= 1
                n += n & 1
                result += 1
            n >>= 1
        return result