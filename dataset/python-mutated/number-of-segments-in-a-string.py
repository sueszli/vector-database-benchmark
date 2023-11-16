class Solution(object):

    def countSegments(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: int\n        '
        result = int(len(s) and s[-1] != ' ')
        for i in xrange(1, len(s)):
            if s[i] == ' ' and s[i - 1] != ' ':
                result += 1
        return result

    def countSegments2(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        return len([i for i in s.strip().split(' ') if i])