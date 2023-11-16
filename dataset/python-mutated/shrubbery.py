@cython.cclass
class Shrubbery:
    width: cython.int
    height: cython.int

    def __init__(self, w, h):
        if False:
            for i in range(10):
                print('nop')
        self.width = w
        self.height = h

    def describe(self):
        if False:
            while True:
                i = 10
        print('This shrubbery is', self.width, 'by', self.height, 'cubits.')