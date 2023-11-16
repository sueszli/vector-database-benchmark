int conflict(int board[][8], int row, int col)
{
    for (int i = 0; i < row; i++) {
        if (board[i][col])
            return 1;
        int j = row - i;
        if (0 < col - j + 1 && board[i][col - j])
            return 1;
        if (col + j < 8 && board[i][col + j])
            return 1;
    }
    return 0;
}

int print_board(int board[][8])
{
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++)
            printf(board[i][j] ? "Q " : ". ");
        printf("\n");
    }
    printf("\n\n");
}

int solve(int board[][8], int row)
{
    if (row == 8) {
        print_board(board);
        return 0;
    }
    for (int i = 0; i < 8; i++) {
        if (!conflict(board, row, i)) {
            board[row][i] = 1;
            solve(board, row + 1);
            board[row][i] = 0;
        }
    }
}

int main()
{
    int board[64];
    for (int i = 0; i < 64; i++)
        board[i] = 0;
    solve(board, 0);
    return 0;
}
