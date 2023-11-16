class Solution(object):

    def waysToBuyPensPencils(self, total, cost1, cost2):
        if False:
            while True:
                i = 10
        '\n        :type total: int\n        :type cost1: int\n        :type cost2: int\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                for i in range(10):
                    print('nop')
            while b:
                (a, b) = (b, a % b)
            return a

        def ceil_divide(a, b):
            if False:
                while True:
                    i = 10
            return (a + b - 1) // b

        def arithmetic_progression_sum(a, d, l):
            if False:
                while True:
                    i = 10
            return (a + (a + (l - 1) * d)) * l // 2
        if cost1 < cost2:
            (cost1, cost2) = (cost2, cost1)
        lcm = cost1 * cost2 // gcd(cost1, cost2)
        result = 0
        d = lcm // cost2
        for i in xrange(min(total // cost1 + 1, lcm // cost1)):
            cnt = (total - i * cost1) // cost2 + 1
            l = ceil_divide(cnt, d)
            result += arithmetic_progression_sum(cnt, -d, l)
        return result

class Solution2(object):

    def waysToBuyPensPencils(self, total, cost1, cost2):
        if False:
            print('Hello World!')
        '\n        :type total: int\n        :type cost1: int\n        :type cost2: int\n        :rtype: int\n        '
        if cost1 < cost2:
            (cost1, cost2) = (cost2, cost1)
        return sum(((total - i * cost1) // cost2 + 1 for i in xrange(total // cost1 + 1)))