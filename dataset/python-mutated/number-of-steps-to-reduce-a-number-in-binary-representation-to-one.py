class Solution(object):

    def numSteps(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: int\n        '
        (result, carry) = (0, 0)
        for i in reversed(xrange(1, len(s))):
            if int(s[i]) + carry == 1:
                carry = 1
                result += 2
            else:
                result += 1
        return result + carry