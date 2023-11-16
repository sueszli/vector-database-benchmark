"""
Unique Paths

Find the unique paths in a matrix starting from the upper left corner and ending in the bottom right corner.

=========================================
Dynamic programming (looking from the left and up neighbour), but this is a slower solution, see the next one.
    Time Complexity:    O(N*M)
    Space Complexity:   O(N*M)
The DP table is creating an Pascal Triangle, so this problem can be easily solved by using the combinatorial formula!
Much faster and doesn't use extra space.
    Time Complexity:    O(min(M, N))
    Space Complexity:   O(1)
"""

def unique_paths_dp(n, m):
    if False:
        for i in range(10):
            print('nop')
    dp = [[1 for j in range(m)] for i in range(n)]
    for i in range(1, n):
        for j in range(1, m):
            dp[i][j] = dp[i][j - 1] + dp[i - 1][j]
    return dp[n - 1][m - 1]

def unique_paths(n, m):
    if False:
        print('Hello World!')
    (m, n) = (min(m, n), max(m, n))
    lvl = m + n - 2
    pos = m - 1
    comb = 1
    for i in range(1, pos + 1):
        comb *= lvl
        comb /= i
        lvl -= 1
    return int(comb + 0.001)
(n, m) = (7, 7)
print(unique_paths(n, m))
print(unique_paths_dp(n, m))
(n, m) = (7, 3)
print(unique_paths(n, m))
print(unique_paths_dp(n, m))
(n, m) = (3, 7)
print(unique_paths(n, m))
print(unique_paths_dp(n, m))