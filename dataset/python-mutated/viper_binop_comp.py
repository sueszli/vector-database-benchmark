@micropython.viper
def f(x: int, y: int):
    if False:
        for i in range(10):
            print('nop')
    if x < y:
        print(x, '<', y)
    if x > y:
        print(x, '>', y)
    if x == y:
        print(x, '==', y)
    if x <= y:
        print(x, '<=', y)
    if x >= y:
        print(x, '>=', y)
    if x != y:
        print(x, '!=', y)
f(1, 1)
f(2, 1)
f(1, 2)
f(2, -1)
f(-2, 1)