class Solution(object):

    def minimumString(self, a, b, c):
        if False:
            i = 10
            return i + 15
        '\n        :type a: str\n        :type b: str\n        :type c: str\n        :rtype: str\n        '

        def getPrefix(pattern):
            if False:
                print('Hello World!')
            prefix = [-1] * len(pattern)
            j = -1
            for i in xrange(1, len(pattern)):
                while j != -1 and pattern[j + 1] != pattern[i]:
                    j = prefix[j]
                if pattern[j + 1] == pattern[i]:
                    j += 1
                prefix[i] = j
            return prefix

        def KMP(text, pattern):
            if False:
                return 10
            prefix = getPrefix(pattern)
            j = -1
            for i in xrange(len(text)):
                while j != -1 and pattern[j + 1] != text[i]:
                    j = prefix[j]
                if pattern[j + 1] == text[i]:
                    j += 1
                if j + 1 == len(pattern):
                    return i - j
            return -1

        def merge(a, b):
            if False:
                print('Hello World!')
            if KMP(b, a) != -1:
                return b
            prefix = getPrefix(b + '#' + a)
            l = prefix[-1] + 1
            return a + b[l:]
        result = [merge(a, merge(b, c)), merge(a, merge(c, b)), merge(b, merge(a, c)), merge(b, merge(c, a)), merge(c, merge(a, b)), merge(c, merge(b, a))]
        return min(result, key=lambda x: (len(x), x))

class Solution2(object):

    def minimumString(self, a, b, c):
        if False:
            print('Hello World!')
        '\n        :type a: str\n        :type b: str\n        :type c: str\n        :rtype: str\n        '

        def merge(a, b):
            if False:
                i = 10
                return i + 15
            if a in b:
                return b
            l = next((l for l in reversed(xrange(1, min(len(a), len(b)))) if a[-l:] == b[:l]), 0)
            return a + b[l:]
        result = [merge(a, merge(b, c)), merge(a, merge(c, b)), merge(b, merge(a, c)), merge(b, merge(c, a)), merge(c, merge(a, b)), merge(c, merge(b, a))]
        return min(result, key=lambda x: (len(x), x))