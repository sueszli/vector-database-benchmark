class Solution:

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        if False:
            i = 10
            return i + 15
        row = [[x for x in y if x != '.'] for y in board]
        col = [[x for x in y if x != '.'] for y in zip(*board)]
        pal = [[board[i + m][j + n] for m in range(3) for n in range(3) if board[i + m][j + n] != '.'] for i in (0, 3, 6) for j in (0, 3, 6)]
        return all((len(set(x)) == len(x) for x in (*row, *col, *pal)))