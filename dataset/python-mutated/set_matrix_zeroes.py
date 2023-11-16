"""
Set Matrix Zeroes

Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in-place.

Input:
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]
Output:
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]

=========================================
Use first column and first row for marking when 0 is found.
    Time Complexity:    O(N*M)
    Space Complexity:   O(1)
"""

def set_matrix_zeroes(matrix):
    if False:
        for i in range(10):
            print('nop')
    n = len(matrix)
    if n == 0:
        return
    m = len(matrix[0])
    is_row = False
    for j in range(m):
        if matrix[0][j] == 0:
            is_row = True
    is_col = False
    for i in range(n):
        if matrix[i][0] == 0:
            is_col = True
    for i in range(1, n):
        for j in range(1, m):
            if matrix[i][j] == 0:
                matrix[i][0] = matrix[0][j] = 0
    for i in range(1, n):
        for j in range(1, m):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    if is_row:
        for j in range(m):
            matrix[0][j] = 0
    if is_col:
        for i in range(n):
            matrix[i][0] = 0
mat = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
set_matrix_zeroes(mat)
print(mat)
mat = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
set_matrix_zeroes(mat)
print(mat)