class Solution(object):

    def superPow(self, a, b):
        if False:
            i = 10
            return i + 15
        '\n        :type a: int\n        :type b: List[int]\n        :rtype: int\n        '

        def myPow(a, n, b):
            if False:
                for i in range(10):
                    print('nop')
            result = 1
            x = a % b
            while n:
                if n & 1:
                    result = result * x % b
                n >>= 1
                x = x * x % b
            return result % b
        result = 1
        for digit in b:
            result = myPow(result, 10, 1337) * myPow(a, digit, 1337) % 1337
        return result