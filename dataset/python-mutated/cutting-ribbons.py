class Solution(object):

    def maxLength(self, ribbons, k):
        if False:
            while True:
                i = 10
        '\n        :type ribbons: List[int]\n        :type k: int\n        :rtype: int\n        '

        def check(ribbons, k, s):
            if False:
                for i in range(10):
                    print('nop')
            return reduce(lambda total, x: total + x // s, ribbons, 0) >= k
        (left, right) = (1, sum(ribbons) // k)
        while left <= right:
            mid = left + (right - left) // 2
            if not check(ribbons, k, mid):
                right = mid - 1
            else:
                left = mid + 1
        return right