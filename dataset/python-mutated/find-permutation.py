class Solution(object):

    def findPermutation(self, s):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :rtype: List[int]\n        '
        result = []
        for i in xrange(len(s) + 1):
            if i == len(s) or s[i] == 'I':
                result += range(i + 1, len(result), -1)
        return result