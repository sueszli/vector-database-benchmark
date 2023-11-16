from sympy.core import Add, Mul, symbols
(x, y, z) = symbols('x,y,z')

def timeit_neg():
    if False:
        print('Hello World!')
    -x

def timeit_Add_x1():
    if False:
        for i in range(10):
            print('nop')
    x + 1

def timeit_Add_1x():
    if False:
        print('Hello World!')
    1 + x

def timeit_Add_x05():
    if False:
        return 10
    x + 0.5

def timeit_Add_xy():
    if False:
        return 10
    x + y

def timeit_Add_xyz():
    if False:
        for i in range(10):
            print('nop')
    Add(*[x, y, z])

def timeit_Mul_xy():
    if False:
        i = 10
        return i + 15
    x * y

def timeit_Mul_xyz():
    if False:
        print('Hello World!')
    Mul(*[x, y, z])

def timeit_Div_xy():
    if False:
        print('Hello World!')
    x / y

def timeit_Div_2y():
    if False:
        i = 10
        return i + 15
    2 / y