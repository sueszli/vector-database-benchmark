from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt

def is_symmetric1(A):
    if False:
        return 10
    return np.all(A == A.T)

def is_symmetric2(A):
    if False:
        while True:
            i = 10
    (rows, cols) = A.shape
    if rows != cols:
        return False
    for i in range(rows):
        for j in range(cols):
            if A[i, j] != A[j, i]:
                return False
    return True

def check(A, b):
    if False:
        for i in range(10):
            print('nop')
    print('Testing:', A)
    assert is_symmetric1(A) == b
    assert is_symmetric2(A) == b
A = np.zeros((3, 3))
check(A, True)
A = np.eye(3)
check(A, True)
A = np.random.randn(3, 2)
A = A.dot(A.T)
check(A, True)
A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
check(A, True)
A = np.random.randn(3, 2)
check(A, False)
A = np.random.randn(3, 3)
check(A, False)
A = np.arange(9).reshape(3, 3)
check(A, False)