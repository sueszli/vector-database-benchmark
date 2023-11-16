"""
Reverse Integer

Given signed integer, reverse digits of an integer.

Input: 123
Output: 321

Input: -123
Output: -321

Input: 120
Output: 21

=========================================
Simple solution, mod 10 to find all digits.
    Time Complexity:    O(N)    , N = number of digits
    Space Complexity:   O(1)
"""

def reverse_integer(x):
    if False:
        while True:
            i = 10
    if x == 0:
        return 0
    sign = x // abs(x)
    x *= sign
    res = 0
    while x > 0:
        res = res * 10 + x % 10
        x //= 10
    return res * sign
print(reverse_integer(123))
print(reverse_integer(-123))
print(reverse_integer(120))