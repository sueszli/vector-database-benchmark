class Solution(object):

    def largestOddNumber(self, num):
        if False:
            i = 10
            return i + 15
        '\n        :type num: str\n        :rtype: str\n        '
        for i in reversed(xrange(len(num))):
            if int(num[i]) % 2:
                return num[:i + 1]
        return ''