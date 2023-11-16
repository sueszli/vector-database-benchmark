class Solution(object):

    def numberOfSteps(self, num):
        if False:
            return 10
        '\n        :type num: int\n        :rtype: int\n        '
        result = 0
        while num:
            result += 2 if num % 2 else 1
            num //= 2
        return max(result - 1, 0)