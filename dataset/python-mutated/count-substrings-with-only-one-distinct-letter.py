class Solution(object):

    def countLetters(self, S):
        if False:
            return 10
        '\n        :type S: str\n        :rtype: int\n        '
        result = len(S)
        left = 0
        for right in xrange(1, len(S)):
            if S[right] == S[left]:
                result += right - left
            else:
                left = right
        return result