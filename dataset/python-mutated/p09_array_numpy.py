"""
Topic: 利用numpy执行数组运算
Desc : 
"""
import numpy as np

def array_numpy():
    if False:
        print('Hello World!')
    x = [1, 2, 3, 4]
    y = [5, 6, 7, 8]
    print(x * 2)
    print(x + y)
    ax = np.array([1, 2, 3, 4])
    ay = np.array([5, 6, 7, 8])
    print(ax * 2)
    print(ax + ay)
    print(ax * ay)
    print(f(ax))
    print(np.sqrt(ax))
    print(np.cos(ax))
    grid = np.zeros(shape=(10000, 10000), dtype=float)
    grid += 10
    print(grid)
    print(np.sin(grid))
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(a)
    print(a[1])
    print(a[:, 1])
    print(a[1:3, 1:3])
    a[1:3, 1:3] += 10
    print(a)
    print(a + [100, 101, 102, 103])
    print(np.where(a < 10, a, 10))

def f(x):
    if False:
        for i in range(10):
            print('nop')
    return 3 * x ** 2 - 2 * x + 7
if __name__ == '__main__':
    array_numpy()