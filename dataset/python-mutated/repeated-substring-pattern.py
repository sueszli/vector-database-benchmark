class Solution(object):

    def repeatedSubstringPattern(self, str):
        if False:
            return 10
        '\n        :type str: str\n        :rtype: bool\n        '

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
        prefix = getPrefix(str)
        return prefix[-1] != -1 and (prefix[-1] + 1) % (len(str) - prefix[-1] - 1) == 0

    def repeatedSubstringPattern2(self, str):
        if False:
            i = 10
            return i + 15
        '\n        :type str: str\n        :rtype: bool\n        '
        if not str:
            return False
        ss = (str + str)[1:-1]
        return ss.find(str) != -1