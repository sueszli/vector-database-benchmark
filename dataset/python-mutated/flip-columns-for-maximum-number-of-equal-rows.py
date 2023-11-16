import collections

class Solution(object):

    def maxEqualRowsAfterFlips(self, matrix):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type matrix: List[List[int]]\n        :rtype: int\n        '
        count = collections.Counter((tuple((x ^ row[0] for x in row)) for row in matrix))
        return max(count.itervalues())