import cython
from cython.cimports.volume import cube

def menu(description, size):
    if False:
        for i in range(10):
            print('nop')
    print(description, ':', cube(size), 'cubic metres of spam')
menu('Entree', 1)
menu('Main course', 3)
menu('Dessert', 2)