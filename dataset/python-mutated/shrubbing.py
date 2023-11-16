import cython

@cython.cclass
class Shrubbery:

    def __cinit__(self, w: cython.int, l: cython.int):
        if False:
            while True:
                i = 10
        self.width = w
        self.length = l

def standard_shrubbery():
    if False:
        i = 10
        return i + 15
    return Shrubbery(3, 7)