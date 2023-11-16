class Solution(object):

    def rotateString(self, A, B):
        if False:
            print('Hello World!')
        '\n        :type A: str\n        :type B: str\n        :rtype: bool\n        '

        def check(index):
            if False:
                i = 10
                return i + 15
            return all((A[(i + index) % len(A)] == c for (i, c) in enumerate(B)))
        if len(A) != len(B):
            return False
        (M, p) = (10 ** 9 + 7, 113)
        p_inv = pow(p, M - 2, M)
        (b_hash, power) = (0, 1)
        for c in B:
            b_hash += power * ord(c)
            b_hash %= M
            power = power * p % M
        (a_hash, power) = (0, 1)
        for i in xrange(len(B)):
            a_hash += power * ord(A[i % len(A)])
            a_hash %= M
            power = power * p % M
        if a_hash == b_hash and check(0):
            return True
        power = power * p_inv % M
        for i in xrange(len(B), 2 * len(A)):
            a_hash = (a_hash - ord(A[(i - len(B)) % len(A)])) * p_inv
            a_hash += power * ord(A[i % len(A)])
            a_hash %= M
            if a_hash == b_hash and check(i - len(B) + 1):
                return True
        return False

class Solution2(object):

    def rotateString(self, A, B):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type A: str\n        :type B: str\n        :rtype: bool\n        '

        def strStr(haystack, needle):
            if False:
                i = 10
                return i + 15

            def KMP(text, pattern):
                if False:
                    while True:
                        i = 10
                prefix = getPrefix(pattern)
                j = -1
                for i in xrange(len(text)):
                    while j > -1 and pattern[j + 1] != text[i]:
                        j = prefix[j]
                    if pattern[j + 1] == text[i]:
                        j += 1
                    if j == len(pattern) - 1:
                        return i - j
                return -1

            def getPrefix(pattern):
                if False:
                    for i in range(10):
                        print('nop')
                prefix = [-1] * len(pattern)
                j = -1
                for i in xrange(1, len(pattern)):
                    while j > -1 and pattern[j + 1] != pattern[i]:
                        j = prefix[j]
                    if pattern[j + 1] == pattern[i]:
                        j += 1
                    prefix[i] = j
                return prefix
            if not needle:
                return 0
            return KMP(haystack, needle)
        if len(A) != len(B):
            return False
        return strStr(A * 2, B) != -1

class Solution3(object):

    def rotateString(self, A, B):
        if False:
            return 10
        '\n        :type A: str\n        :type B: str\n        :rtype: bool\n        '
        return len(A) == len(B) and B in A * 2