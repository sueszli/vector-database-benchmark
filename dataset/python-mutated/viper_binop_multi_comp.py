@micropython.viper
def f(x: int, y: int):
    if False:
        for i in range(10):
            print('nop')
    if 0 < x < y:
        print(x, '<', y)
    if 3 > x > y:
        print(x, '>', y)
    if 1 == x == y:
        print(x, '==', y)
    if -2 == x <= y:
        print(x, '<=', y)
    if 2 == x >= y:
        print(x, '>=', y)
    if 2 == x != y:
        print(x, '!=', y)
f(1, 1)
f(2, 1)
f(1, 2)
f(2, -1)
f(-2, 1)