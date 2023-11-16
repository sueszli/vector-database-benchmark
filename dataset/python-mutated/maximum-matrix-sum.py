class Solution(object):

    def maxMatrixSum(self, matrix):
        if False:
            while True:
                i = 10
        '\n        :type matrix: List[List[int]]\n        :rtype: int\n        '
        abs_total = sum((abs(x) for row in matrix for x in row))
        min_abs_val = min((abs(x) for row in matrix for x in row))
        neg_cnt = sum((x < 0 for row in matrix for x in row))
        return abs_total if neg_cnt % 2 == 0 else abs_total - 2 * min_abs_val