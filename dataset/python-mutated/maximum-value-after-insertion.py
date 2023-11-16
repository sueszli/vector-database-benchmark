class Solution(object):

    def maxValue(self, n, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: str\n        :type x: int\n        :rtype: str\n        '
        check = (lambda i: str(x) > n[i]) if n[0] != '-' else lambda i: str(x) < n[i]
        for i in xrange(len(n)):
            if check(i):
                break
        else:
            i = len(n)
        return n[:i] + str(x) + n[i:]