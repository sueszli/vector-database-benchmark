class Solution(object):

    def minOperations(self, nums, x, y):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type x: int\n        :type y: int\n        :rtype: int\n        '

        def ceil_divide(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return (a + b - 1) // b

        def check(total):
            if False:
                for i in range(10):
                    print('nop')
            return sum((ceil_divide(max(v - min(ceil_divide(v, y), total) * y, 0), x - y) for v in nums)) <= total
        (left, right) = (1, ceil_divide(max(nums), y))
        while left <= right:
            mid = left + (right - left) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return left