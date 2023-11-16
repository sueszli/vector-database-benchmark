class Solution(object):

    def smallestString(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: str\n        '
        result = list(s)
        i = next((i for i in xrange(len(s)) if s[i] != 'a'), len(s))
        if i == len(s):
            result[-1] = 'z'
        else:
            for i in xrange(i, len(s)):
                if result[i] == 'a':
                    break
                result[i] = chr(ord(result[i]) - 1)
        return ''.join(result)