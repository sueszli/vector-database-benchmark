class Solution(object):

    def shortestPalindrome(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: str\n        '

        def getPrefix(pattern):
            if False:
                print('Hello World!')
            prefix = [-1] * len(pattern)
            j = -1
            for i in xrange(1, len(pattern)):
                while j > -1 and pattern[j + 1] != pattern[i]:
                    j = prefix[j]
                if pattern[j + 1] == pattern[i]:
                    j += 1
                prefix[i] = j
            return prefix
        if not s:
            return s
        A = s + '#' + s[::-1]
        return s[getPrefix(A)[-1] + 1:][::-1] + s

class Solution2(object):

    def shortestPalindrome(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: str\n        '

        def getPrefix(pattern):
            if False:
                print('Hello World!')
            prefix = [-1] * len(pattern)
            j = -1
            for i in xrange(1, len(pattern)):
                while j > -1 and pattern[j + 1] != pattern[i]:
                    j = prefix[j]
                if pattern[j + 1] == pattern[i]:
                    j += 1
                prefix[i] = j
            return prefix
        if not s:
            return s
        A = s + s[::-1]
        prefix = getPrefix(A)
        i = prefix[-1]
        while i >= len(s):
            i = prefix[i]
        return s[i + 1:][::-1] + s

class Solution3(object):

    def shortestPalindrome(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: str\n        '

        def preProcess(s):
            if False:
                for i in range(10):
                    print('nop')
            if not s:
                return ['^', '$']
            string = ['^']
            for c in s:
                string += ['#', c]
            string += ['#', '$']
            return string
        string = preProcess(s)
        palindrome = [0] * len(string)
        (center, right) = (0, 0)
        for i in xrange(1, len(string) - 1):
            i_mirror = 2 * center - i
            if right > i:
                palindrome[i] = min(right - i, palindrome[i_mirror])
            else:
                palindrome[i] = 0
            while string[i + 1 + palindrome[i]] == string[i - 1 - palindrome[i]]:
                palindrome[i] += 1
            if i + palindrome[i] > right:
                (center, right) = (i, i + palindrome[i])
        max_len = 0
        for i in xrange(1, len(string) - 1):
            if i - palindrome[i] == 1:
                max_len = palindrome[i]
        return s[len(s) - 1:max_len - 1:-1] + s