import cython

@cython.cclass
class A:
    cython.declare(a=cython.int, b=cython.int)
    c = cython.declare(cython.int, visibility='public')
    d = cython.declare(cython.int)
    e = cython.declare(cython.int, visibility='readonly')

    def __init__(self, a, b, c, d=5, e=3):
        if False:
            return 10
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e