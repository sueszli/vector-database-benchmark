import collections

class Solution(object):

    def longestPalindrome(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        odds = 0
        for (k, v) in collections.Counter(s).iteritems():
            odds += v & 1
        return len(s) - odds + int(odds > 0)

    def longestPalindrome2(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: int\n        '
        odd = sum(map(lambda x: x & 1, collections.Counter(s).values()))
        return len(s) - odd + int(odd > 0)