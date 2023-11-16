class Solution(object):

    def integerReplacement(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: int\n        '
        result = 0
        while n != 1:
            b = n & 3
            if n == 3:
                n -= 1
            elif b == 3:
                n += 1
            elif b == 1:
                n -= 1
            else:
                n /= 2
            result += 1
        return result

class Solution2(object):

    def integerReplacement(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :rtype: int\n        '
        if n < 4:
            return [0, 0, 1, 2][n]
        if n % 4 in (0, 2):
            return self.integerReplacement(n / 2) + 1
        elif n % 4 == 1:
            return self.integerReplacement((n - 1) / 4) + 3
        else:
            return self.integerReplacement((n + 1) / 4) + 3