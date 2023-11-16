@cython.cclass
class Function:

    @cython.ccall
    def evaluate(self, x: float) -> float:
        if False:
            return 10
        return 0