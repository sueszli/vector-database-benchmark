class Solution(object):

    def solveSudoku(self, board):
        if False:
            while True:
                i = 10

        def isValid(board, x, y):
            if False:
                for i in range(10):
                    print('nop')
            for i in xrange(9):
                if i != x and board[i][y] == board[x][y]:
                    return False
            for j in xrange(9):
                if j != y and board[x][j] == board[x][y]:
                    return False
            i = 3 * (x / 3)
            while i < 3 * (x / 3 + 1):
                j = 3 * (y / 3)
                while j < 3 * (y / 3 + 1):
                    if (i != x or j != y) and board[i][j] == board[x][y]:
                        return False
                    j += 1
                i += 1
            return True

        def solver(board):
            if False:
                print('Hello World!')
            for i in xrange(len(board)):
                for j in xrange(len(board[0])):
                    if board[i][j] == '.':
                        for k in xrange(9):
                            board[i][j] = chr(ord('1') + k)
                            if isValid(board, i, j) and solver(board):
                                return True
                            board[i][j] = '.'
                        return False
            return True
        solver(board)