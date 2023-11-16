class Solution(object):

    def kthPalindrome(self, queries, intLength):
        if False:
            print('Hello World!')
        '\n        :type queries: List[int]\n        :type intLength: int\n        :rtype: List[int]\n        '

        def reverse(x):
            if False:
                return 10
            result = 0
            while x:
                result = result * 10 + x % 10
                x //= 10
            return result

        def f(l, x):
            if False:
                print('Hello World!')
            x = 10 ** ((l - 1) // 2) + (x - 1)
            if x > 10 ** ((l + 1) // 2) - 1:
                return -1
            return x * 10 ** (l // 2) + reverse(x // 10 if l % 2 else x)
        return [f(intLength, x) for x in queries]

class Solution2(object):

    def kthPalindrome(self, queries, intLength):
        if False:
            i = 10
            return i + 15
        '\n        :type queries: List[int]\n        :type intLength: int\n        :rtype: List[int]\n        '

        def f(l, x):
            if False:
                for i in range(10):
                    print('nop')
            if 10 ** ((l - 1) // 2) + (x - 1) > 10 ** ((l + 1) // 2) - 1:
                return -1
            s = str(10 ** ((l - 1) // 2) + (x - 1))
            return int(s + s[::-1][l % 2:])
        return [f(intLength, x) for x in queries]