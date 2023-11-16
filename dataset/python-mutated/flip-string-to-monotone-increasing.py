class Solution(object):

    def minFlipsMonoIncr(self, S):
        if False:
            print('Hello World!')
        '\n        :type S: str\n        :rtype: int\n        '
        (flip0, flip1) = (0, 0)
        for c in S:
            flip0 += int(c == '1')
            flip1 = min(flip0, flip1 + int(c == '0'))
        return flip1