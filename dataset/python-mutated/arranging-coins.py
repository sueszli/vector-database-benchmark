import math

class Solution(object):

    def arrangeCoins(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: int\n        '
        return int((math.sqrt(8 * n + 1) - 1) / 2)

class Solution2(object):

    def arrangeCoins(self, n):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :rtype: int\n        '

        def check(mid, n):
            if False:
                for i in range(10):
                    print('nop')
            return mid * (mid + 1) <= 2 * n
        (left, right) = (1, n)
        while left <= right:
            mid = left + (right - left) // 2
            if not check(mid, n):
                right = mid - 1
            else:
                left = mid + 1
        return right