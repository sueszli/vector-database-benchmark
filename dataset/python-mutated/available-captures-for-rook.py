class Solution(object):

    def numRookCaptures(self, board):
        if False:
            while True:
                i = 10
        '\n        :type board: List[List[str]]\n        :rtype: int\n        '
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        (r, c) = (None, None)
        for i in xrange(8):
            if r is not None:
                break
            for j in xrange(8):
                if board[i][j] == 'R':
                    (r, c) = (i, j)
                    break
        result = 0
        for d in directions:
            (nr, nc) = (r + d[0], c + d[1])
            while 0 <= nr < 8 and 0 <= nc < 8:
                if board[nr][nc] == 'p':
                    result += 1
                if board[nr][nc] != '.':
                    break
                (nr, nc) = (nr + d[0], nc + d[1])
        return result