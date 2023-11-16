@micropython.native
def native_test(x):
    if False:
        print('Hello World!')
    print(1, [], x)
native_test(2)
import gc
gc.collect()
native_test(3)

@micropython.native
def f(a, b):
    if False:
        return 10
    print(a + b)
f(1, 2)

@micropython.native
def f(a, b, c):
    if False:
        return 10
    print(a + b + c)
f(1, 2, 3)

@micropython.native
def f(a):
    if False:
        return 10
    print(not a)
f(False)
f(True)

@micropython.native
def f(a):
    if False:
        while True:
            i = 10
    print(1, 2, 3, 4 if a else 5)
f(False)
f(True)