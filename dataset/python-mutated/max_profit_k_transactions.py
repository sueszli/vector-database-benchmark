"""
Max Profit With K Transactions

You are given an array of integers representing the prices of a single stock on various days
(each index in the array represents a different day).
You are also given an integer k, which represents the number of transactions you are allowed to make.
One transaction consists of buying the stock on a given day and selling it on another, later day.
Write a function that returns the maximum profit that you can make buying and selling the stock,
given k transactions. Note that you can only hold 1 share of the stock at a time; in other words,
you cannot buy more than 1 share of the stock on any given day, and you cannot buy a share of the
stock if you are still holding another share.
In a day, you can first sell a share and buy another after that.

Input: [5, 11, 3, 50, 60, 90], 2
Output: 93
Output explanation: Buy 5, Sell 11; Buy 3, Sell 90

=========================================
Optimized dynamic programming solution.
For this solution you'll need only the current and previous rows.
The original (not optimized) DP formula is: MAX(dp[t][d-1], price[d] + MAX(dp[t-1][x] - price[x])),
but this is O(K * N^2) Time Complexity, and O(N * K) space complexity.
    Time Complexity:    O(N * К)
    Space Complexity:   O(N)
"""
import math

def max_profit_with_k_transactions(prices, k):
    if False:
        for i in range(10):
            print('nop')
    days = len(prices)
    if days < 2:
        return 0
    k = min(k, days)
    dp = [[0 for j in range(days)] for i in range(2)]
    for t in range(k):
        max_prev = -math.inf
        prev_idx = (t - 1) % 2
        curr_idx = t % 2
        past_days = t
        dp[curr_idx][past_days] = dp[prev_idx][past_days]
        for d in range(past_days + 1, days):
            max_prev = max(max_prev, dp[prev_idx][d - 1] - prices[d - 1])
            dp[curr_idx][d] = max(dp[curr_idx][d - 1], max_prev + prices[d])
    return dp[(k - 1) % 2][-1]
print(max_profit_with_k_transactions([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10))
print(max_profit_with_k_transactions([5, 11, 3, 50, 60, 90], 2))