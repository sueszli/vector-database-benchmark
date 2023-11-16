"""
Topic: 矩阵和线性代数
Desc : 
"""
import numpy as np
import numpy.linalg

def matrix_linear():
    if False:
        return 10
    m = np.matrix([[1, -2, 3], [0, 4, 5], [7, 8, -9]])
    print(m)
    print(m.T)
    print(m.I)
    v = np.matrix([[2], [3], [4]])
    print(v)
    print(m * v)
    print(numpy.linalg.det(m))
    print(numpy.linalg.eigvals(m))
    x = numpy.linalg.solve(m, v)
    print(x)
    print(m * x)
    print(v)
if __name__ == '__main__':
    matrix_linear()