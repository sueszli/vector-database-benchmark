@micropython.viper
def f(x: uint, y: uint):
    if False:
        return 10
    if x < y:
        print(' <', end='')
    if x > y:
        print(' >', end='')
    if x == y:
        print(' ==', end='')
    if x <= y:
        print(' <=', end='')
    if x >= y:
        print(' >=', end='')
    if x != y:
        print(' !=', end='')

def test(a, b):
    if False:
        return 10
    print(a, b, end='')
    f(a, b)
    print()
test(1, 1)
test(2, 1)
test(1, 2)
test(2, -1)
test(-2, 1)
test(-2, -1)