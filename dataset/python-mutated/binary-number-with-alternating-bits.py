class Solution(object):

    def hasAlternatingBits(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: bool\n        '
        (n, curr) = divmod(n, 2)
        while n > 0:
            if curr == n % 2:
                return False
            (n, curr) = divmod(n, 2)
        return True