import itertools

class Solution(object):

    def luckyNumbers(self, matrix):
        if False:
            print('Hello World!')
        '\n        :type matrix: List[List[int]]\n        :rtype: List[int]\n        '
        rows = map(min, matrix)
        cols = map(max, itertools.izip(*matrix))
        return [cell for (i, row) in enumerate(matrix) for (j, cell) in enumerate(row) if rows[i] == cols[j]]
import itertools

class Solution2(object):

    def luckyNumbers(self, matrix):
        if False:
            while True:
                i = 10
        '\n        :type matrix: List[List[int]]\n        :rtype: List[int]\n        '
        return list(set(map(min, matrix)) & set(map(max, itertools.izip(*matrix))))