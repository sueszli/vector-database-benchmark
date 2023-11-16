class Solution(object):

    def minimizedMaximum(self, n, quantities):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type quantities: List[int]\n        :rtype: int\n        '

        def ceil_divide(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return (a + (b - 1)) // b

        def check(n, quantities, x):
            if False:
                i = 10
                return i + 15
            return sum((ceil_divide(q, x) for q in quantities)) <= n
        (left, right) = (1, max(quantities))
        while left <= right:
            mid = left + (right - left) // 2
            if check(n, quantities, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left