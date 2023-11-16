class Solution(object):

    def nthMagicalNumber(self, N, A, B):
        if False:
            i = 10
            return i + 15
        '\n        :type N: int\n        :type A: int\n        :type B: int\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                for i in range(10):
                    print('nop')
            while b:
                (a, b) = (b, a % b)
            return a

        def check(A, B, N, lcm, target):
            if False:
                while True:
                    i = 10
            return target // A + target // B - target // lcm >= N
        lcm = A * B // gcd(A, B)
        (left, right) = (min(A, B), max(A, B) * N)
        while left <= right:
            mid = left + (right - left) // 2
            if check(A, B, N, lcm, mid):
                right = mid - 1
            else:
                left = mid + 1
        return left % (10 ** 9 + 7)