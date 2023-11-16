class Solution(object):

    def findKthNumber(self, m, n, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type m: int\n        :type n: int\n        :type k: int\n        :rtype: int\n        '

        def count(target, m, n):
            if False:
                for i in range(10):
                    print('nop')
            return sum((min(target // i, n) for i in xrange(1, m + 1)))
        (left, right) = (1, m * n)
        while left <= right:
            mid = left + (right - left) / 2
            if count(mid, m, n) >= k:
                right = mid - 1
            else:
                left = mid + 1
        return left