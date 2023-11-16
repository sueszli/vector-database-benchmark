class Solution(object):

    def countOperations(self, num1, num2):
        if False:
            while True:
                i = 10
        '\n        :type num1: int\n        :type num2: int\n        :rtype: int\n        '
        result = 0
        while num2:
            result += num1 // num2
            (num1, num2) = (num2, num1 % num2)
        return result