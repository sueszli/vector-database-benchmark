import cython

@cython.freelist(8)
@cython.cclass
class Penguin:
    food: object

    def __cinit__(self, food):
        if False:
            while True:
                i = 10
        self.food = food
penguin = Penguin('fish 1')
penguin = None
penguin = Penguin('fish 2')