class Solution(object):

    def sumBase(self, n, k):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type k: int\n        :rtype: int\n        '
        result = 0
        while n:
            (n, r) = divmod(n, k)
            result += r
        return result