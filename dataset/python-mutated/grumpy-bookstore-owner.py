class Solution(object):

    def maxSatisfied(self, customers, grumpy, X):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type customers: List[int]\n        :type grumpy: List[int]\n        :type X: int\n        :rtype: int\n        '
        (result, max_extra, extra) = (0, 0, 0)
        for i in xrange(len(customers)):
            result += 0 if grumpy[i] else customers[i]
            extra += customers[i] if grumpy[i] else 0
            if i >= X:
                extra -= customers[i - X] if grumpy[i - X] else 0
            max_extra = max(max_extra, extra)
        return result + max_extra