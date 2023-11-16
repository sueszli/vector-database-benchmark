class Solution(object):

    def convert(self, s, numRows):
        if False:
            return 10
        '\n        :type s: str\n        :type numRows: int\n        :rtype: str\n        '
        if numRows == 1:
            return s
        (step, zigzag) = (2 * numRows - 2, '')
        for i in xrange(numRows):
            for j in xrange(i, len(s), step):
                zigzag += s[j]
                if 0 < i < numRows - 1 and j + step - 2 * i < len(s):
                    zigzag += s[j + step - 2 * i]
        return zigzag