def pacific_atlantic(matrix):
    if False:
        i = 10
        return i + 15
    '\n    :type matrix: List[List[int]]\n    :rtype: List[List[int]]\n    '
    n = len(matrix)
    if not n:
        return []
    m = len(matrix[0])
    if not m:
        return []
    res = []
    atlantic = [[False for _ in range(n)] for _ in range(m)]
    pacific = [[False for _ in range(n)] for _ in range(m)]
    for i in range(n):
        dfs(pacific, matrix, float('-inf'), i, 0)
        dfs(atlantic, matrix, float('-inf'), i, m - 1)
    for i in range(m):
        dfs(pacific, matrix, float('-inf'), 0, i)
        dfs(atlantic, matrix, float('-inf'), n - 1, i)
    for i in range(n):
        for j in range(m):
            if pacific[i][j] and atlantic[i][j]:
                res.append([i, j])
    return res

def dfs(grid, matrix, height, i, j):
    if False:
        i = 10
        return i + 15
    if i < 0 or i >= len(matrix) or j < 0 or (j >= len(matrix[0])):
        return
    if grid[i][j] or matrix[i][j] < height:
        return
    grid[i][j] = True
    dfs(grid, matrix, matrix[i][j], i - 1, j)
    dfs(grid, matrix, matrix[i][j], i + 1, j)
    dfs(grid, matrix, matrix[i][j], i, j - 1)
    dfs(grid, matrix, matrix[i][j], i, j + 1)