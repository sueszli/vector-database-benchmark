from cython.cimports.sin_of_square import Function

@cython.cclass
class WaveFunction(Function):
    offset: float
    freq = cython.declare(cython.double, visibility='public')
    scale = cython.declare(cython.double, visibility='readonly')

    @property
    def period(self):
        if False:
            return 10
        return 1.0 / self.freq

    @period.setter
    def period(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.freq = 1.0 / value