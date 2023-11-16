class Solution(object):

    def checkMove(self, board, rMove, cMove, color):
        if False:
            i = 10
            return i + 15
        '\n        :type board: List[List[str]]\n        :type rMove: int\n        :type cMove: int\n        :type color: str\n        :rtype: bool\n        '

        def check(board, color, r, c, dr, dc):
            if False:
                for i in range(10):
                    print('nop')
            l = 2
            while 0 <= r < len(board) and 0 <= c < len(board[0]) and (board[r][c] != '.'):
                if board[r][c] == color:
                    return l >= 3
                r += dr
                c += dc
                l += 1
            return False
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1)]
        for (dr, dc) in directions:
            (r, c) = (rMove + dr, cMove + dc)
            if check(board, color, r, c, dr, dc):
                return True
        return False