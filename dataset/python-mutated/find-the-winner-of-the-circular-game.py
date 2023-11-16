class Solution(object):

    def findTheWinner(self, n, k):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type k: int\n        :rtype: int\n        '
        return reduce(lambda idx, n: (idx + k) % (n + 1), xrange(1, n), 0) + 1

class Solution2(object):

    def findTheWinner(self, n, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type k: int\n        :rtype: int\n        '

        def f(idx, n, k):
            if False:
                return 10
            if n == 1:
                return 0
            return (k + f((idx + k) % n, n - 1, k)) % n
        return f(0, n, k) + 1