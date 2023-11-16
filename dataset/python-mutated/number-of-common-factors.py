class Solution(object):

    def commonFactors(self, a, b):
        if False:
            print('Hello World!')
        '\n        :type a: int\n        :type b: int\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                return 10
            while b:
                (a, b) = (b, a % b)
            return a
        g = gcd(a, b)
        result = 0
        x = 1
        while x * x <= g:
            if g % x == 0:
                result += 1 if g // x == x else 2
            x += 1
        return result