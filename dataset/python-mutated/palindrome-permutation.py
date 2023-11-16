import collections

class Solution(object):

    def canPermutePalindrome(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :rtype: bool\n        '
        return sum((v % 2 for v in collections.Counter(s).values())) < 2