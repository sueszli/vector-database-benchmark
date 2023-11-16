class Solution(object):

    def removePalindromeSub(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '

        def is_palindrome(s):
            if False:
                i = 10
                return i + 15
            for i in xrange(len(s) // 2):
                if s[i] != s[-1 - i]:
                    return False
            return True
        return 2 - is_palindrome(s) - (s == '')