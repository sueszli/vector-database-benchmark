class Solution(object):

    def sumOfNumberAndReverse(self, num):
        if False:
            print('Hello World!')
        '\n        :type num: int\n        :rtype: bool\n        '

        def backtracking(num, chosen):
            if False:
                while True:
                    i = 10
            if num == 0:
                return True
            if chosen == 1:
                return False
            if num <= 18:
                return num % 2 == 0 or (num == 11 and chosen == 0)
            if chosen == 2:
                return False
            for x in (num % 10, 10 + num % 10):
                if not 1 <= x <= 18:
                    continue
                base = 11
                if chosen:
                    base = chosen
                else:
                    while x * ((base - 1) * 10 + 1) <= num:
                        base = (base - 1) * 10 + 1
                if num - x * base >= 0 and backtracking((num - x * base) // 10, base // 100 + 1):
                    return True
            return False
        return backtracking(num, 0)

class Solution2(object):

    def sumOfNumberAndReverse(self, num):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type num: int\n        :rtype: bool\n        '

        def reverse(n):
            if False:
                print('Hello World!')
            result = 0
            while n:
                result = result * 10 + n % 10
                n //= 10
            return result
        return any((x + reverse(x) == num for x in xrange(num // 2, num + 1)))

class Solution3(object):

    def sumOfNumberAndReverse(self, num):
        if False:
            return 10
        '\n        :type num: int\n        :rtype: bool\n        '
        return any((x + int(str(x)[::-1]) == num for x in xrange(num // 2, num + 1)))