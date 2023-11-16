class Solution(object):

    def gcdOfStrings(self, str1, str2):
        if False:
            i = 10
            return i + 15
        '\n        :type str1: str\n        :type str2: str\n        :rtype: str\n        '

        def check(s, common):
            if False:
                while True:
                    i = 10
            i = 0
            for c in s:
                if c != common[i]:
                    return False
                i = (i + 1) % len(common)
            return True

        def gcd(a, b):
            if False:
                while True:
                    i = 10
            while b:
                (a, b) = (b, a % b)
            return a
        if not str1 or not str2:
            return ''
        c = gcd(len(str1), len(str2))
        result = str1[:c]
        return result if check(str1, result) and check(str2, result) else ''