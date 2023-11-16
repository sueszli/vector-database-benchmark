class Solution(object):

    def pivotInteger(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: int\n        '
        x = int(((n + 1) * n // 2) ** 0.5 + 0.5)
        return x if x ** 2 == (n + 1) * n // 2 else -1