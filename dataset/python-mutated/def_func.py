import math

def my_abs(x):
    if False:
        i = 10
        return i + 15
    if not isinstance(x, (int, float)):
        raise TypeError('bad operand type')
    if x >= 0:
        return x
    else:
        return -x

def move(x, y, step, angle=0):
    if False:
        while True:
            i = 10
    nx = x + step * math.cos(angle)
    ny = y - step * math.sin(angle)
    return (nx, ny)
n = my_abs(-20)
print(n)
(x, y) = move(100, 100, 60, math.pi / 6)
print(x, y)
my_abs('123')