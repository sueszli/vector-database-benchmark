class Solution(object):

    def subtractProductAndSum(self, n):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :rtype: int\n        '
        (product, total) = (1, 0)
        while n:
            (n, r) = divmod(n, 10)
            product *= r
            total += r
        return product - total
import operator

class Solution2(object):

    def subtractProductAndSum(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: int\n        '
        A = map(int, str(n))
        return reduce(operator.mul, A) - sum(A)