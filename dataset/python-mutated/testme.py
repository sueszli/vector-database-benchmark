import numpy as np
arr = [i for i in range(1, 1000)]

def doit1(x):
    if False:
        while True:
            i = 10
    y = 1
    x = [i * i for i in range(0, 100000)][99999]
    y1 = [i * i for i in range(0, 200000)][199999]
    z1 = [i for i in range(0, 300000)][299999]
    z = x * y
    return z

def doit2(x):
    if False:
        print('Hello World!')
    i = 0
    z = 0.1
    while i < 100000:
        z = z * z
        z = x * x
        z = z * z
        z = z * z
        i += 1
    return z

def doit3(x):
    if False:
        for i in range(10):
            print('nop')
    z = x + 1
    z = x + 1
    z = x + 1
    z = x + z
    z = x + z
    return z

def stuff():
    if False:
        print('Hello World!')
    x = 1.01
    for i in range(1, 10):
        print(i)
        for j in range(1, 10):
            x = doit1(x)
            x = doit2(x)
            x = doit3(x)
            x = 1.01
    return x
import sys
print('TESTME')
print(sys.argv)
stuff()