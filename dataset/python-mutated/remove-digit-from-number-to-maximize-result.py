class Solution(object):

    def removeDigit(self, number, digit):
        if False:
            i = 10
            return i + 15
        '\n        :type number: str\n        :type digit: str\n        :rtype: str\n        '
        i = next((i for i in xrange(len(number) - 1) if digit == number[i] < number[i + 1]), len(number) - 1)
        if i + 1 == len(number):
            i = next((i for i in reversed(xrange(len(number))) if digit == number[i]))
        return number[:i] + number[i + 1:]