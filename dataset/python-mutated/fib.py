"""
In mathematics, the Fibonacci numbers, commonly denoted Fn,
form a sequence, called the Fibonacci sequence,
such that each number is the sum of the two preceding ones,
starting from 0 and 1.
That is,
    F0=0 , F1=1
and
    Fn= F(n-1) + F(n-2)
The Fibonacci numbers are the numbers in the following integer sequence.
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, …….

In mathematical terms, the sequence Fn of Fibonacci numbers is
defined by the recurrence relation

Here, given a number n, print n-th Fibonacci Number.
"""

def fib_recursive(n):
    if False:
        return 10
    '[summary]\n    Computes the n-th fibonacci number recursive.\n    Problem: This implementation is very slow.\n    approximate O(2^n)\n\n    Arguments:\n        n {[int]} -- [description]\n\n    Returns:\n        [int] -- [description]\n    '
    assert n >= 0, 'n must be a positive integer'
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)

def fib_list(n):
    if False:
        print('Hello World!')
    '[summary]\n    This algorithm computes the n-th fibbonacci number\n    very quick. approximate O(n)\n    The algorithm use dynamic programming.\n\n    Arguments:\n        n {[int]} -- [description]\n\n    Returns:\n        [int] -- [description]\n    '
    assert n >= 0, 'n must be a positive integer'
    list_results = [0, 1]
    for i in range(2, n + 1):
        list_results.append(list_results[i - 1] + list_results[i - 2])
    return list_results[n]

def fib_iter(n):
    if False:
        return 10
    '[summary]\n    Works iterative approximate O(n)\n\n    Arguments:\n        n {[int]} -- [description]\n\n    Returns:\n        [int] -- [description]\n    '
    assert n >= 0, 'n must be positive integer'
    fib_1 = 0
    fib_2 = 1
    res = 0
    if n <= 1:
        return n
    for _ in range(n - 1):
        res = fib_1 + fib_2
        fib_1 = fib_2
        fib_2 = res
    return res