import numpy as np

class A:

    def __init__(self, n):
        if False:
            print('Hello World!')
        self.arr = np.random.rand(n)
        self.lst = [1] * n
        print(n)
if __name__ == '__main__':
    a = A(50000000)