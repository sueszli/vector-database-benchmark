class Solution(object):

    def minimizeSet(self, divisor1, divisor2, uniqueCnt1, uniqueCnt2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type divisor1: int\n        :type divisor2: int\n        :type uniqueCnt1: int\n        :type uniqueCnt2: int\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                return 10
            while b:
                (a, b) = (b, a % b)
            return a

        def lcm(a, b):
            if False:
                print('Hello World!')
            return a // gcd(a, b) * b

        def count(cnt, d1, d2):
            if False:
                for i in range(10):
                    print('nop')
            l = lcm(d1, d2)
            return cnt + cnt // (l - 1) - int(cnt % (l - 1) == 0)
        return max(count(uniqueCnt1, divisor1, 1), count(uniqueCnt2, divisor2, 1), count(uniqueCnt1 + uniqueCnt2, divisor1, divisor2))

class Solution2(object):

    def minimizeSet(self, divisor1, divisor2, uniqueCnt1, uniqueCnt2):
        if False:
            while True:
                i = 10
        '\n        :type divisor1: int\n        :type divisor2: int\n        :type uniqueCnt1: int\n        :type uniqueCnt2: int\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                for i in range(10):
                    print('nop')
            while b:
                (a, b) = (b, a % b)
            return a

        def lcm(a, b):
            if False:
                while True:
                    i = 10
            return a // gcd(a, b) * b

        def check(cnt):
            if False:
                while True:
                    i = 10
            return cnt - cnt // divisor1 >= uniqueCnt1 and cnt - cnt // divisor2 >= uniqueCnt2 and (cnt - cnt // l >= uniqueCnt1 + uniqueCnt2)
        l = lcm(divisor1, divisor2)
        (left, right) = (2, 2 ** 31 - 1)
        while left <= right:
            mid = left + (right - left) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return left