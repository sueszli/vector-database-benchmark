class Solution(object):

    def oddCells(self, n, m, indices):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type m: int\n        :type indices: List[List[int]]\n        :rtype: int\n        '
        (row, col) = ([0] * n, [0] * m)
        for (r, c) in indices:
            row[r] ^= 1
            col[c] ^= 1
        (row_sum, col_sum) = (sum(row), sum(col))
        return row_sum * m + col_sum * n - 2 * row_sum * col_sum
import collections
import itertools

class Solution2(object):

    def oddCells(self, n, m, indices):
        if False:
            return 10
        '\n        :type n: int\n        :type m: int\n        :type indices: List[List[int]]\n        :rtype: int\n        '
        fn = lambda x: sum((count & 1 for count in collections.Counter(x).itervalues()))
        (row_sum, col_sum) = map(fn, itertools.izip(*indices))
        return row_sum * m + col_sum * n - 2 * row_sum * col_sum