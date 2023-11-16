class Solution(object):

    def champagneTower(self, poured, query_row, query_glass):
        if False:
            return 10
        '\n        :type poured: int\n        :type query_row: int\n        :type query_glass: int\n        :rtype: float\n        '
        result = [poured] + [0] * query_row
        for i in xrange(1, query_row + 1):
            for j in reversed(xrange(i + 1)):
                result[j] = max(result[j] - 1, 0) / 2.0 + max(result[j - 1] - 1, 0) / 2.0
        return min(result[query_glass], 1)