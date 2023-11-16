"""
Functions to calculate nCr (ie how many ways to choose r items from n items)
"""

def combination(n, r):
    if False:
        for i in range(10):
            print('nop')
    'This function calculates nCr.'
    if n == r or r == 0:
        return 1
    return combination(n - 1, r - 1) + combination(n - 1, r)

def combination_memo(n, r):
    if False:
        for i in range(10):
            print('nop')
    'This function calculates nCr using memoization method.'
    memo = {}

    def recur(n, r):
        if False:
            for i in range(10):
                print('nop')
        if n == r or r == 0:
            return 1
        if (n, r) not in memo:
            memo[n, r] = recur(n - 1, r - 1) + recur(n - 1, r)
        return memo[n, r]
    return recur(n, r)