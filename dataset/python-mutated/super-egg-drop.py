class Solution(object):

    def superEggDrop(self, K, N):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type K: int\n        :type N: int\n        :rtype: int\n        '

        def check(n, K, N):
            if False:
                i = 10
                return i + 15
            (total, c) = (0, 1)
            for k in xrange(1, K + 1):
                c *= n - k + 1
                c //= k
                total += c
                if total >= N:
                    return True
            return False
        (left, right) = (1, N)
        while left <= right:
            mid = left + (right - left) // 2
            if check(mid, K, N):
                right = mid - 1
            else:
                left = mid + 1
        return left