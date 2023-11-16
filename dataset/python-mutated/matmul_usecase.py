"""Use cases for testing matmul (@)
"""

def matmul_usecase(x, y):
    if False:
        for i in range(10):
            print('nop')
    return x @ y

def imatmul_usecase(x, y):
    if False:
        while True:
            i = 10
    x @= y
    return x

class DumbMatrix(object):

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self.value = value

    def __matmul__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, DumbMatrix):
            return DumbMatrix(self.value * other.value)
        return NotImplemented

    def __imatmul__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, DumbMatrix):
            self.value *= other.value
            return self
        return NotImplemented