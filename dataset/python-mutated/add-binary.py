class Solution(object):

    def addBinary(self, a, b):
        if False:
            return 10
        (result, carry, val) = ('', 0, 0)
        for i in xrange(max(len(a), len(b))):
            val = carry
            if i < len(a):
                val += int(a[-(i + 1)])
            if i < len(b):
                val += int(b[-(i + 1)])
            (carry, val) = divmod(val, 2)
            result += str(val)
        if carry:
            result += str(carry)
        return result[::-1]
from itertools import izip_longest

class Solution2(object):

    def addBinary(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type a: str\n        :type b: str\n        :rtype: str\n        '
        result = ''
        carry = 0
        for (x, y) in izip_longest(reversed(a), reversed(b), fillvalue='0'):
            (carry, remainder) = divmod(int(x) + int(y) + carry, 2)
            result += str(remainder)
        if carry:
            result += str(carry)
        return result[::-1]