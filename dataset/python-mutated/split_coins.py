"""
Split Coins

You have a number of coins with various amounts.
You need to split the coins in two groups so that the difference between those groups in minimal.

Input: [1, 1, 1, 3, 5, 10, 18]
Output: 1
Output explanation: First group 1, 3, 5, 10 (or 1, 1, 3, 5, 10) and second group 1, 1, 18 (or 1, 18).

=========================================
Simple dynamic programming solution. Find the closest sum to the half of the sum of all coins.
    Time Complexity:    O(C*HS)     , C = number of coins, HS = half of the sum of all coins
    Space Complexity:   O(HS)
"""

def split_coins(coins):
    if False:
        while True:
            i = 10
    if len(coins) == 0:
        return -1
    full_sum = sum(coins)
    half_sum = full_sum // 2 + 1
    dp = [False] * half_sum
    dp[0] = True
    for c in coins:
        for i in range(half_sum - 1, -1, -1):
            if i >= c and dp[i - c]:
                dp[i] = True
    for i in range(half_sum - 1, -1, -1):
        if dp[i]:
            return full_sum - 2 * i
    return -1
print(split_coins([1, 1, 1, 3, 5, 10, 18]))