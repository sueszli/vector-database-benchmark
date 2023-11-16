class Solution(object):

    def breakPalindrome(self, palindrome):
        if False:
            print('Hello World!')
        '\n        :type palindrome: str\n        :rtype: str\n        '
        for i in xrange(len(palindrome) // 2):
            if palindrome[i] != 'a':
                return palindrome[:i] + 'a' + palindrome[i + 1:]
        return palindrome[:-1] + 'b' if len(palindrome) >= 2 else ''