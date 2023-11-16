class Solution(object):

    def replaceDigits(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: str\n        '
        return ''.join((chr(ord(s[i - 1]) + int(s[i])) if i % 2 else s[i] for i in xrange(len(s))))