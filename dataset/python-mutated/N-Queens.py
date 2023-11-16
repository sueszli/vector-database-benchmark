global N
N = 4

def printSolution(board):
    if False:
        print('Hello World!')
    for i in range(N):
        for j in range(N):
            print(board[i][j], end=' ')
        print()

def isSafe(board, row, col):
    if False:
        return 10
    for i in range(col):
        if board[row][i] == 1:
            return False
    for (i, j) in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    for (i, j) in zip(range(row, N, 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    return True

def solveNQUtil(board, col):
    if False:
        print('Hello World!')
    if col >= N:
        return True
    for i in range(N):
        if isSafe(board, i, col):
            board[i][col] = 1
            if solveNQUtil(board, col + 1) == True:
                return True
            board[i][col] = 0
    return False

def solveNQ():
    if False:
        while True:
            i = 10
    board = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    if solveNQUtil(board, 0) == False:
        print('Solution does not exist')
        return False
    printSolution(board)
    return True
solveNQ()