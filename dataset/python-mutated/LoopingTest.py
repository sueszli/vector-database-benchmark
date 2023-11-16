""" Looping in various forms.

"""
from __future__ import print_function

def cond():
    if False:
        for i in range(10):
            print('nop')
    return False

def loopingFunction(a=1 * 2):
    if False:
        while True:
            i = 10
    c = []
    f = [c, a]
    for a in range(6 or 8):
        for b in range(8):
            if a == b:
                c.append((a, b, True))
            elif a < b:
                c.append((b, a, False))
            else:
                c.append((a, b, False))
            if a != b:
                z = 1
            else:
                z = 0
            if z == 0:
                continue
            if z == 1 and b == 6:
                break
            if a == b:
                z = 0
    print(c)
    print(f)
    f = 1
    while f < (10 or 8):
        m = 1
        f += 1
    print('m=', m)
    x = [u for u in range(8)]
    print(x)
    x = [(u, v) for (u, v) in zip(range(8), reversed(range(8)))]
    print(x)
    x = [u if u % 2 == 0 else 0 for u in range(10)]
    print(x)
    x = [u if u % 2 == 0 else 0 for u in (a if cond() else range(9))]
    print(x)
    y = [[3 + (l if l else -1) for l in [m, m + 1]] for m in [f for f in range(2)]]
    print('f=', f)
    print('y=', y)
    if x:
        l = 'YES'
    else:
        l = 'NO'
    if x:
        l = 'yes'
    elif True:
        l = 'no'
    print('Triple and chain')
    if m and l and f:
        print('OK')
    print('Triple or chain')
    if m or l or f:
        print('Okey')
    print('Nested if not chain')
    if not m:
        if not l:
            print('ok')
    print("Braced if not chain with 'or'")
    if not (m or l):
        print('oki')
    print("Braced if not chain with 'and'")
    if not (m and l):
        print('oki')
    d = 1
    print('Nested if chain with outer else')
    if a:
        if b or c:
            if d:
                print('inside nest')
    else:
        print('outer else')
    print(x)
    while False:
        pass
    else:
        print('Executed else branch for False condition while loop')
    while True:
        break
    else:
        print('Executed else branch for True condition while loop')
    for x in range(7):
        pass
    else:
        print('Executed else branch for no break for loop')
    for x in range(7):
        break
    else:
        print('Executed else branch despite break in for loop')
    x = iter(range(5))
    while next(x):
        pass
    else:
        print('Executed else branch of while loop without break')
loopingFunction()