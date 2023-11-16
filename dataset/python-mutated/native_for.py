@micropython.native
def f1(n):
    if False:
        for i in range(10):
            print('nop')
    for i in range(n):
        print(i)
f1(4)

@micropython.native
def f2(r):
    if False:
        while True:
            i = 10
    for i in r:
        print(i)
f2(range(4))