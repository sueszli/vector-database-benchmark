import cython

def func():
    if False:
        print('Hello World!')
    x: cython.int
    y: cython.double = 0.57721
    z: cython.float = 0.57721
    a: float = 0.54321
    b: int = 5
    c: long = 6
    pass

@cython.cclass
class A:
    a: cython.int
    b: cython.int

    def __init__(self, b=0):
        if False:
            return 10
        self.a = 3
        self.b = b