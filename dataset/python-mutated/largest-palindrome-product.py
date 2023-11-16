class Solution(object):

    def largestPalindrome(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: int\n        '
        if n == 1:
            return 9
        upper = 10 ** n - 1
        for k in xrange(2, upper + 1):
            left = 10 ** n - k
            right = int(str(left)[::-1])
            d = k ** 2 - right * 4
            if d < 0:
                continue
            if d ** 0.5 == int(d ** 0.5) and k % 2 == int(d ** 0.5) % 2:
                return (left * 10 ** n + right) % 1337
        return -1

class Solution2(object):

    def largestPalindrome(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :rtype: int\n        '

        def divide_ceil(a, b):
            if False:
                return 10
            return (a + b - 1) // b
        if n == 1:
            return 9
        (upper, lower) = (10 ** n - 1, 10 ** (n - 1))
        for i in reversed(xrange(lower, upper ** 2 // 10 ** n + 1)):
            candidate = int(str(i) + str(i)[::-1])
            for y in reversed(xrange(divide_ceil(lower, 11) * 11, upper + 1, 11)):
                if candidate // y > upper:
                    break
                if candidate % y == 0 and lower <= candidate // y:
                    return candidate % 1337
        return -1