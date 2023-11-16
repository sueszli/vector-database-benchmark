@micropython.native
def f(n):
    if False:
        i = 10
        return i + 15
    i = 0
    while i < n:
        print(i)
        i += 1
f(2)
f(4)