class Solution(object):

    def checkPalindromeFormation(self, a, b):
        if False:
            print('Hello World!')
        '\n        :type a: str\n        :type b: str\n        :rtype: bool\n        '

        def is_palindrome(s, i, j):
            if False:
                for i in range(10):
                    print('nop')
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True

        def check(a, b):
            if False:
                return 10
            (i, j) = (0, len(b) - 1)
            while i < j:
                if a[i] != b[j]:
                    return is_palindrome(a, i, j) or is_palindrome(b, i, j)
                i += 1
                j -= 1
            return True
        return check(a, b) or check(b, a)