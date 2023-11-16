class Solution(object):

    def nthUglyNumber(self, n, a, b, c):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type a: int\n        :type b: int\n        :type c: int\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                i = 10
                return i + 15
            while b:
                (a, b) = (b, a % b)
            return a

        def lcm(x, y):
            if False:
                print('Hello World!')
            return x // gcd(x, y) * y

        def count(x, a, b, c, lcm_a_b, lcm_b_c, lcm_c_a, lcm_a_b_c):
            if False:
                for i in range(10):
                    print('nop')
            return x // a + x // b + x // c - (x // lcm_a_b + x // lcm_b_c + x // lcm_c_a) + x // lcm_a_b_c
        (lcm_a_b, lcm_b_c, lcm_c_a) = (lcm(a, b), lcm(b, c), lcm(c, a))
        lcm_a_b_c = lcm(lcm_a_b, lcm_b_c)
        (left, right) = (1, 2 * 10 ** 9)
        while left <= right:
            mid = left + (right - left) // 2
            if count(mid, a, b, c, lcm_a_b, lcm_b_c, lcm_c_a, lcm_a_b_c) >= n:
                right = mid - 1
            else:
                left = mid + 1
        return left