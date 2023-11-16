"""
Spiral Matrix

Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.

Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output: [1, 2, 3, 6, 9, 8, 7, 4, 5]

Input:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12]
]
Output: [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]

=========================================
Simulate spiral moving, start from (0,0) and when a border is reached change the X or Y direction.
    Time Complexity:    O(N*M)
    Space Complexity:   O(N*M)
"""

def spiral_matrix(matrix):
    if False:
        print('Hello World!')
    n = len(matrix)
    if n == 0:
        return []
    m = len(matrix[0])
    if m == 0:
        return []
    total = n * m
    res = []
    n -= 1
    (xDir, yDir) = (1, 1)
    (x, y) = (0, -1)
    while len(res) < total:
        for i in range(m):
            y += yDir
            res.append(matrix[x][y])
        m -= 1
        yDir *= -1
        for i in range(n):
            x += xDir
            res.append(matrix[x][y])
        n -= 1
        xDir *= -1
    return res
print(spiral_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
print(spiral_matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))