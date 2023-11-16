from sympy.core.numbers import Integer
from sympy.matrices.dense import eye, zeros
i3 = Integer(3)
M = eye(100)

def timeit_Matrix__getitem_ii():
    if False:
        while True:
            i = 10
    M[3, 3]

def timeit_Matrix__getitem_II():
    if False:
        for i in range(10):
            print('nop')
    M[i3, i3]

def timeit_Matrix__getslice():
    if False:
        print('Hello World!')
    M[:, :]

def timeit_Matrix_zeronm():
    if False:
        for i in range(10):
            print('nop')
    zeros(100, 100)