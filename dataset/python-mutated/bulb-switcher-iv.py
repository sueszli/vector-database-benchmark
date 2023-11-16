class Solution(object):

    def minFlips(self, target):
        if False:
            while True:
                i = 10
        '\n        :type target: str\n        :rtype: int\n        '
        (result, curr) = (0, '0')
        for c in target:
            if c == curr:
                continue
            curr = c
            result += 1
        return result