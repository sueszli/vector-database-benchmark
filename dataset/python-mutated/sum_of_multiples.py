"""
Multiples of a OR b

If we list all the natural numbers below 10 that are multiples of 3 OR 5, we get 3, 5, 6 and 9.
The sum of these multiples is 23.
Find the sum of all the multiples of A or B below N.

=========================================
Don't need iteration to solve this problem, you need to find only how many divisors are there.
Example - 3 + 6 + 9 ... + N = (1 + 2 + 3 + ... N // 3) * 3
Sum(K)*N = 1*N + 2*N + ... + (K-1)*N + K*N
Use sum formula - (N * (N + 1))/2
    Time Complexity:    O(1)
    Space Complexity:   O(1)
"""

def sum_of_multiples_below(a, b, total):
    if False:
        print('Hello World!')
    total -= 1
    return sum_of_dividends(total, a) + sum_of_dividends(total, b) - sum_of_dividends(total, a * b)

def sum_of_dividends(total, divisor):
    if False:
        i = 10
        return i + 15
    n = total // divisor
    return n * (n + 1) // 2 * divisor
print(sum_of_multiples_below(3, 5, 10))
print(sum_of_multiples_below(3, 5, 1000))