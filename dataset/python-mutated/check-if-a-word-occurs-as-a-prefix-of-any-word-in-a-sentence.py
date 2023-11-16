class Solution(object):

    def isPrefixOfWord(self, sentence, searchWord):
        if False:
            return 10
        '\n        :type sentence: str\n        :type searchWord: str\n        :rtype: int\n        '

        def KMP(text, pattern):
            if False:
                for i in range(10):
                    print('nop')

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
        if sentence.startswith(searchWord):
            return 1
        p = KMP(sentence, ' ' + searchWord)
        if p == -1:
            return -1
        return 1 + sum((sentence[i] == ' ' for i in xrange(p + 1)))