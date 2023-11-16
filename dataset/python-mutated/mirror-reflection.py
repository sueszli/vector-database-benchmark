class Solution(object):

    def mirrorReflection(self, p, q):
        if False:
            i = 10
            return i + 15
        '\n        :type p: int\n        :type q: int\n        :rtype: int\n        '
        return 2 if p & -p > q & -q else 0 if p & -p < q & -q else 1

class Solution2(object):

    def mirrorReflection(self, p, q):
        if False:
            print('Hello World!')
        '\n        :type p: int\n        :type q: int\n        :rtype: int\n        '

        def gcd(a, b):
            if False:
                while True:
                    i = 10
            while b:
                (a, b) = (b, a % b)
            return a
        lcm = p * q // gcd(p, q)
        if lcm // p % 2 == 1:
            if lcm // q % 2 == 1:
                return 1
            return 2
        return 0