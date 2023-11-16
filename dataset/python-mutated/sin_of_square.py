from cython.cimports.libc.math import sin

@cython.cclass
class Function:

    @cython.ccall
    def evaluate(self, x: float) -> float:
        if False:
            print('Hello World!')
        return 0

@cython.cclass
class SinOfSquareFunction(Function):

    @cython.ccall
    def evaluate(self, x: float) -> float:
        if False:
            i = 10
            return i + 15
        return sin(x ** 2)