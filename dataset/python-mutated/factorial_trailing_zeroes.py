"""
Factorial Trailing Zeroes

Given an integer n, return the number of trailing zeroes in n!.

Note: Your solution should be in logarithmic time complexity.

Input: 3
Output: 0
Output explanation: 3! = 6, no trailing zero.

Input: 5
Output: 1
Output explanation: 5! = 120, one trailing zero.

=========================================
Find how many 5s are in range 0-N (more explanation in the solution).
    Time Complexity:    O(logN)
    Space Complexity:   O(1)
"""

def trailing_zeroes(n):
    if False:
        for i in range(10):
            print('nop')
    res = 0
    k = 5
    while k <= n:
        res += n // k
        k *= 5
    return res