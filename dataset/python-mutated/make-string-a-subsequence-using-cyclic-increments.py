class Solution(object):

    def canMakeSubsequence(self, str1, str2):
        if False:
            while True:
                i = 10
        '\n        :type str1: str\n        :type str2: str\n        :rtype: bool\n        '
        i = 0
        for c in str1:
            if (ord(str2[i]) - ord(c)) % 26 > 1:
                continue
            i += 1
            if i == len(str2):
                return True
        return False