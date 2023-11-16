class Solution(object):

    def isValidSudoku(self, board):
        if False:
            while True:
                i = 10
        '\n        :type board: List[List[str]]\n        :rtype: bool\n        '
        for i in xrange(9):
            if not self.isValidList([board[i][j] for j in xrange(9)]) or not self.isValidList([board[j][i] for j in xrange(9)]):
                return False
        for i in xrange(3):
            for j in xrange(3):
                if not self.isValidList([board[m][n] for n in xrange(3 * j, 3 * j + 3) for m in xrange(3 * i, 3 * i + 3)]):
                    return False
        return True

    def isValidList(self, xs):
        if False:
            return 10
        xs = filter(lambda x: x != '.', xs)
        return len(set(xs)) == len(xs)