"""
Min Cost Coloring

A builder is looking to build a row of N houses that can be of K different colors.
He has a goal of minimizing cost while ensuring that no two neighboring houses are of the same color.
Given an N by K matrix where the nth row and kth column represents the cost to build the
nth house with kth color, return the minimum cost which achieves this goal.

=========================================
Dynamic programming, for each house search for the cheapest combination of the previous houses.
But don't search the whole array with combinations (colors), save only the smallest 2
(in this case we're sure that the previous house doesn't have the same color).
    Time Complexity:    O(N * K)
    Space Complexity:   O(1)
"""
import math

def min_cost_coloring(dp):
    if False:
        i = 10
        return i + 15
    n = len(dp)
    if n == 0:
        return 0
    m = len(dp[0])
    if m < 2:
        return -1
    prev_min = [(0, -1), (0, -1)]
    for i in range(n):
        curr_min = [(math.inf, -1), (math.inf, -1)]
        for j in range(m):
            if j != prev_min[0][1]:
                dp[i][j] += prev_min[0][0]
            else:
                dp[i][j] += prev_min[1][0]
            if curr_min[0][0] > dp[i][j]:
                curr_min[1] = curr_min[0]
                curr_min[0] = (dp[i][j], j)
            elif curr_min[1][0] > dp[i][j]:
                curr_min[1] = (dp[i][j], j)
        prev_min = curr_min
    return min(dp[n - 1])
print(min_cost_coloring([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [3, 2, 1, 4, 5], [3, 2, 1, 4, 3]]))
print(min_cost_coloring([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]))