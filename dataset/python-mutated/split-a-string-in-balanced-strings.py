class Solution(object):

    def balancedStringSplit(self, s):
        if False:
            return 10
        '\n        :type s: str\n        :rtype: int\n        '
        (result, count) = (0, 0)
        for c in s:
            count += 1 if c == 'L' else -1
            if count == 0:
                result += 1
        return result