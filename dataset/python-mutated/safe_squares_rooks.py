"""
Safe Squares from Rooks

On a generalized n-by-n chessboard, there are some number of rooks, each rook represented as a
two-tuple (row, column) of the row and the column that it is in. (The rows and columns are
numbered from 0 to n-1.) A chess rook covers all squares that are in the same row or in the same
column as that rook. Given the board size n and the list of rooks on that board, count the number of
empty squares that are safe, that is, are not covered by any rook.

Input: [(1, 1), (3, 5), (7, 0), (7, 6)], 8
Output: 20

=========================================
The result is a multiplication between free rows and free columns.
Use hashsets to store the free rows and columns.
    Time Complexity:    O(N)
    Space Complexity:   O(N)
"""

def safe_squares_rooks(rooks, n):
    if False:
        print('Hello World!')
    rows = set()
    cols = set()
    for i in range(n):
        rows.add(i)
        cols.add(i)
    for rook in rooks:
        if rook[0] in rows:
            rows.remove(rook[0])
        if rook[1] in cols:
            cols.remove(rook[1])
    return len(rows) * len(cols)
print(safe_squares_rooks([(1, 1)], 2))
print(safe_squares_rooks([(2, 3), (0, 1)], 4))
print(safe_squares_rooks([(1, 1), (3, 5), (7, 0), (7, 6)], 8))
print(safe_squares_rooks([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)], 6))