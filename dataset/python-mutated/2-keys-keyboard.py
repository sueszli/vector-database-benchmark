class Solution(object):

    def minSteps(self, n):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :rtype: int\n        '
        result = 0
        p = 2
        while p ** 2 <= n:
            while n % p == 0:
                result += p
                n //= p
            p += 1
        if n > 1:
            result += n
        return result