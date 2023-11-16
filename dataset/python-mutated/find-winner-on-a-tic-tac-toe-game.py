class Solution(object):

    def tictactoe(self, moves):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type moves: List[List[int]]\n        :rtype: str\n        '
        (row, col) = ([[0] * 3 for _ in xrange(2)], [[0] * 3 for _ in xrange(2)])
        (diag, anti_diag) = ([0] * 2, [0] * 2)
        p = 0
        for (r, c) in moves:
            row[p][r] += 1
            col[p][c] += 1
            diag[p] += r == c
            anti_diag[p] += r + c == 2
            if 3 in (row[p][r], col[p][c], diag[p], anti_diag[p]):
                return 'AB'[p]
            p ^= 1
        return 'Draw' if len(moves) == 9 else 'Pending'