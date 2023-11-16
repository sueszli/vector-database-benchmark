class Solution(object):

    def fixedRatio(self, s, num1, num2):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :type num1: int\n        :type num2: int\n        :rtype: int\n        '
        lookup = collections.Counter()
        lookup[0] = 1
        result = curr = 0
        for c in s:
            curr += -num2 if c == '0' else +num1
            result += lookup[curr]
            lookup[curr] += 1
        return result