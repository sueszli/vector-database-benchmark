class Solution(object):

    def countEven(self, num):
        if False:
            return 10
        '\n        :type num: int\n        :rtype: int\n        '

        def parity(x):
            if False:
                print('Hello World!')
            result = 0
            while x:
                result += x % 10
                x //= 10
            return result % 2
        return (num - parity(num)) // 2

class Solution2(object):

    def countEven(self, num):
        if False:
            return 10
        '\n        :type num: int\n        :rtype: int\n        '

        def parity(x):
            if False:
                for i in range(10):
                    print('nop')
            result = 0
            while x:
                result += x % 10
                x //= 10
            return result % 2
        return sum((parity(x) == 0 for x in xrange(1, num + 1)))

class Solution3(object):

    def countEven(self, num):
        if False:
            i = 10
            return i + 15
        '\n        :type num: int\n        :rtype: int\n        '
        return sum((sum(map(int, str(x))) % 2 == 0 for x in xrange(1, num + 1)))