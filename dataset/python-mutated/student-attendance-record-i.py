class Solution(object):

    def checkRecord(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: bool\n        '
        count_A = 0
        for i in xrange(len(s)):
            if s[i] == 'A':
                count_A += 1
                if count_A == 2:
                    return False
            if i < len(s) - 2 and s[i] == s[i + 1] == s[i + 2] == 'L':
                return False
        return True