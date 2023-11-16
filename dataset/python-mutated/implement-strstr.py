class Solution(object):

    def strStr(self, haystack, needle):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type haystack: str\n        :type needle: str\n        :rtype: int\n        '
        if not needle:
            return 0
        return self.KMP(haystack, needle)

    def KMP(self, text, pattern):
        if False:
            for i in range(10):
                print('nop')
        prefix = self.getPrefix(pattern)
        j = -1
        for i in xrange(len(text)):
            while j > -1 and pattern[j + 1] != text[i]:
                j = prefix[j]
            if pattern[j + 1] == text[i]:
                j += 1
            if j == len(pattern) - 1:
                return i - j
        return -1

    def getPrefix(self, pattern):
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

class Solution2(object):

    def strStr(self, haystack, needle):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type haystack: str\n        :type needle: str\n        :rtype: int\n        '
        for i in xrange(len(haystack) - len(needle) + 1):
            if haystack[i:i + len(needle)] == needle:
                return i
        return -1