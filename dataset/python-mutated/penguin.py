import cython

@cython.cclass
class Penguin:
    food: object

    def __cinit__(self, food):
        if False:
            print('Hello World!')
        self.food = food

    def __init__(self, food):
        if False:
            i = 10
            return i + 15
        print('eating!')
normal_penguin = Penguin('fish')
fast_penguin = Penguin.__new__(Penguin, 'wheat')