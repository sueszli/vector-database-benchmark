"""A Dynamic Programming solution for Rod cutting problem
"""
INT_MIN = -32767

def cut_rod(price):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the best obtainable price for a rod of length n and\n    price[] as prices of different pieces\n    '
    n = len(price)
    val = [0] * (n + 1)
    for i in range(1, n + 1):
        max_val = INT_MIN
        for j in range(i):
            max_val = max(max_val, price[j] + val[i - j - 1])
        val[i] = max_val
    return val[n]
arr = [1, 5, 8, 9, 10, 17, 17, 20]
print('Maximum Obtainable Value is ' + str(cut_rod(arr)))