class Solution(object):

    def validPalindrome(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: bool\n        '

        def validPalindrome(s, left, right):
            if False:
                for i in range(10):
                    print('nop')
            while left < right:
                if s[left] != s[right]:
                    return False
                (left, right) = (left + 1, right - 1)
            return True
        (left, right) = (0, len(s) - 1)
        while left < right:
            if s[left] != s[right]:
                return validPalindrome(s, left, right - 1) or validPalindrome(s, left + 1, right)
            (left, right) = (left + 1, right - 1)
        return True