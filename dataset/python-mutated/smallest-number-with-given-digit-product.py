class Solution(object):

    def smallestNumber(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: str\n        '
        result = []
        for d in reversed(xrange(2, 9 + 1)):
            while n % d == 0:
                result.append(d)
                n //= d
        return ''.join(map(str, reversed(result))) or '1' if n == 1 else '-1'