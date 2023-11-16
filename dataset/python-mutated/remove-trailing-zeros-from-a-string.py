class Solution(object):

    def removeTrailingZeros(self, num):
        if False:
            while True:
                i = 10
        '\n        :type num: str\n        :rtype: str\n        '
        return num[:next((i for i in reversed(xrange(len(num))) if num[i] != '0')) + 1]