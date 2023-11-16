class Solution(object):

    def makeSmallestPalindrome(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: str\n        '
        return ''.join((min(s[i], s[~i]) for i in xrange(len(s))))