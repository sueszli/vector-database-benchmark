class Solution(object):

    def maxValue(self, n, index, maxSum):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type index: int\n        :type maxSum: int\n        :rtype: int\n        '

        def check(n, index, maxSum, x):
            if False:
                for i in range(10):
                    print('nop')
            y = max(x - index, 0)
            total = (x + y) * (x - y + 1) // 2
            y = max(x - (n - 1 - index), 0)
            total += (x + y) * (x - y + 1) // 2
            return total - x <= maxSum
        maxSum -= n
        (left, right) = (0, maxSum)
        while left <= right:
            mid = left + (right - left) // 2
            if not check(n, index, maxSum, mid):
                right = mid - 1
            else:
                left = mid + 1
        return 1 + right