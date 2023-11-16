import re

class Solution(object):

    def fractionAddition(self, expression):
        if False:
            i = 10
            return i + 15
        '\n        :type expression: str\n        :rtype: str\n        '

        def gcd(a, b):
            if False:
                print('Hello World!')
            while b:
                (a, b) = (b, a % b)
            return a
        ints = map(int, re.findall('[+-]?\\d+', expression))
        (A, B) = (0, 1)
        for i in xrange(0, len(ints), 2):
            (a, b) = (ints[i], ints[i + 1])
            A = A * b + a * B
            B *= b
            g = gcd(A, B)
            A //= g
            B //= g
        return '%d/%d' % (A, B)