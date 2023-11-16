"""Test: annotated global."""
n: int

def f():
    if False:
        for i in range(10):
            print('nop')
    print(n)

def g():
    if False:
        i = 10
        return i + 15
    global n
    n = 1
g()
f()