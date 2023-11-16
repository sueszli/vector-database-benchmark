"""
Transform Number Ascending Digits

Given a number and we need to transform to a new number where all its digits are ordered in a non descending order.
All digits can be increased, decreased, over/underflow are allowed.
Find the minimum number of operations we need to do to create a new number with its ordered digits.

Input: '5982'
Output: 4
Output explanation: 5999, 1 operation to transform 8 to 9, 3 operations to transform 2 to 9.

=========================================
Dynamic programming solution.
    Time Complexity:    O(N)    , O(N * 10 * 10) = O(100 N) = O(N)
    Space Complexity:   O(1)    , O(10 * 10) = O(100) = O(1)
"""

def operations(number):
    if False:
        for i in range(10):
            print('nop')
    n = len(number)
    diff = lambda i, j: abs(j - int(number[i]))
    prev_dp = [min(diff(0, i), 10 - diff(0, i)) for i in range(10)]
    for i in range(1, n):
        curr_dp = [min(diff(i, j), 10 - diff(i, j)) for j in range(10)]
        for j in range(10):
            curr_dp[j] += min(prev_dp[0:j + 1])
        prev_dp = curr_dp
    min_dist = min(prev_dp)
    return min_dist
print(operations('901'))
print(operations('301'))
print(operations('5982'))