"""
Should emit:
B002 - on lines 18, 19, and 24
"""

def this_is_all_fine(n):
    if False:
        for i in range(10):
            print('nop')
    x = n + 1
    y = 1 + n
    z = +x + y
    a = n - 1
    b = 1 - n
    c = -a - b
    return (+z, -c)

def this_is_buggy(n):
    if False:
        while True:
            i = 10
    x = ++n
    y = --n
    return (x, y)

def this_is_buggy_too(n):
    if False:
        return 10
    return (++n, --n)