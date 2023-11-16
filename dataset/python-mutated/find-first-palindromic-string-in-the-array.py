class Solution(object):

    def firstPalindrome(self, words):
        if False:
            while True:
                i = 10
        '\n        :type words: List[str]\n        :rtype: str\n        '

        def is_palindrome(s):
            if False:
                while True:
                    i = 10
            (i, j) = (0, len(s) - 1)
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True
        for w in words:
            if is_palindrome(w):
                return w
        return ''

class Solution2(object):

    def firstPalindrome(self, words):
        if False:
            print('Hello World!')
        '\n        :type words: List[str]\n        :rtype: str\n        '
        return next((x for x in words if x == x[::-1]), '')