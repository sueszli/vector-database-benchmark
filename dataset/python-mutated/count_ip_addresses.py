"""
Count IP Addresses

An IP Address (IPv4) consists of 4 numbers which are all between 0 and 255.
In this problem however, we are dealing with 'Extended IP Addresses' which consist of K such numbers.
Given a string S containing only digits and a number K,
your task is to count how many valid 'Extended IP Addresses' can be formed.
An Extended IP Address is valid if:
* it consists of exactly K numbers
* each numbers is between 0 and 255, inclusive
* a number cannot have leading zeroes

Input: '1234567', 3
Output: 1
Output explanation: Valid IP addresses: '123.45.67'.

Input: '100111', 3
Output: 1
Output explanation: Valid IP addresses: '100.1.11', '100.11.1', '10.0.111'.

Input: '345678', 2
Output: 0
Output explanation: It is not possible to form a valid IP Address with two numbers.

=========================================
1D Dynamic programming solution.
    Time Complexity:    O(N*K)
    Space Complexity:   O(N)
"""

def count_ip_addresses(S, K):
    if False:
        i = 10
        return i + 15
    n = len(S)
    if n == 0:
        return 0
    if n < K:
        return 0
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(K):
        for j in range(n, i, -1):
            dp[j] = 0
            for e in range(max(i, j - 3), j):
                if is_valid(S[e:j]):
                    dp[j] += dp[e]
    return dp[n]

def is_valid(S):
    if False:
        while True:
            i = 10
    if len(S) > 1 and S[0] == '0':
        return False
    return int(S) <= 255
print(count_ip_addresses('1234567', 3))
print(count_ip_addresses('100111', 3))
print(count_ip_addresses('345678', 2))