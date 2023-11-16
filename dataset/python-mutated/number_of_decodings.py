"""
Number of Decodings

Given the mapping a=1, b=2, ... , z=26, and an encoded message, count the number of ways it can be decoded.
For example, the message "111" would give 3, since it could be decoded as "aaa", "ka" and "ak".
All of the messages are decodable!

=========================================
The easiest solution is Brute-Force (building a tree and making all combinations),
and in the worst case there will be Fibbionaci(N) combinations, so the worst Time Complexity will be O(Fib(N))

Dynamic programming solution. Similar to number_of_smses.py.
    Time Complexity:    O(N)
    Space Complexity:   O(N)
"""

def num_decodings(code):
    if False:
        for i in range(10):
            print('nop')
    n = len(code)
    dp = [0 for i in range(n)]
    if n == 0:
        return 0
    dp[0] = 1
    if n == 1:
        return dp[0]
    dp[1] = (code[1] != '0') + is_valid(code[0:2])
    for i in range(2, n):
        if code[i] != '0':
            dp[i] += dp[i - 1]
        if is_valid(code[i - 1:i + 1]):
            dp[i] += dp[i - 2]
    return dp[n - 1]

def is_valid(code):
    if False:
        for i in range(10):
            print('nop')
    k = int(code)
    return k < 27 and k > 9
print(num_decodings('12151'))
print(num_decodings('1111'))
print(num_decodings('111'))
print(num_decodings('1010'))
print(num_decodings('2626'))
print(num_decodings('1'))
print(num_decodings('11'))
print(num_decodings('111'))
print(num_decodings('1111'))
print(num_decodings('11111'))
print(num_decodings('111111'))
print(num_decodings('1111111'))
print(num_decodings('11111111'))