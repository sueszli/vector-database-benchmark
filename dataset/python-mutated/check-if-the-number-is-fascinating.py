class Solution(object):

    def isFascinating(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: bool\n        '
        lookup = [0]

        def check(x):
            if False:
                print('Hello World!')
            while x:
                (x, d) = divmod(x, 10)
                if d == 0 or lookup[0] & 1 << d:
                    return False
                lookup[0] |= 1 << d
            return True
        return check(n) and check(2 * n) and check(3 * n)

class Solution2(object):

    def isFascinating(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :rtype: bool\n        '
        s = str(n) + str(2 * n) + str(3 * n)
        return '0' not in s and len(s) == 9 and (len(set(s)) == 9)