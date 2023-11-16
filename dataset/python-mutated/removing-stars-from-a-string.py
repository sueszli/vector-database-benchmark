class Solution(object):

    def removeStars(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: str\n        '
        result = []
        for c in s:
            if c == '*':
                result.pop()
            else:
                result.append(c)
        return ''.join(result)