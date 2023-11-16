@cython.cclass
class Animal:
    number_of_legs: cython.int

    def __cinit__(self, number_of_legs: cython.int):
        if False:
            print('Hello World!')
        self.number_of_legs = number_of_legs

class ExtendableAnimal(Animal):
    pass
dog = ExtendableAnimal(4)
dog.has_tail = True