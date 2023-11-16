class Solution(object):

    def gameOfLife(self, board):
        if False:
            print('Hello World!')
        '\n        :type board: List[List[int]]\n        :rtype: void Do not return anything, modify board in-place instead.\n        '
        m = len(board)
        n = len(board[0]) if m else 0
        for i in xrange(m):
            for j in xrange(n):
                count = 0
                for I in xrange(max(i - 1, 0), min(i + 2, m)):
                    for J in xrange(max(j - 1, 0), min(j + 2, n)):
                        count += board[I][J] & 1
                if count == 4 and board[i][j] or count == 3:
                    board[i][j] |= 2
        for i in xrange(m):
            for j in xrange(n):
                board[i][j] >>= 1