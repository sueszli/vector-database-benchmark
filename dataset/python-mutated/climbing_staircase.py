"""
Climbing Staircase

There exists a staircase with N steps, and you can climb up either X different steps at a time.
Given N, write a function that returns the number of unique ways you can climb the staircase.
The order of the steps matters.

Input: steps = [1, 2], height = 4
Output: 5
Output explanation:
1, 1, 1, 1
2, 1, 1
1, 2, 1
1, 1, 2
2, 2

=========================================
Dynamic Programing solution.
    Time Complexity:    O(N*S)
    Space Complexity:   O(N)
"""

def climbing_staircase(steps, height):
    if False:
        while True:
            i = 10
    dp = [0 for i in range(height)]
    for s in steps:
        if s <= height:
            dp[s - 1] = 1
    for i in range(height):
        for s in steps:
            if i - s >= 0:
                dp[i] += dp[i - s]
    return dp[height - 1]
print(climbing_staircase([1, 2], 4))
print(climbing_staircase([1, 3, 5], 4))