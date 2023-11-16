@micropython.viper
def add(x: uint, y: uint):
    if False:
        for i in range(10):
            print('nop')
    return (x + y, y + x)
print('add')
print(*add(1, 2))
print(*(x & 4294967295 for x in add(-1, -2)))

@micropython.viper
def sub(x: uint, y: uint):
    if False:
        return 10
    return (x - y, y - x)
print('sub')
print(*(x & 4294967295 for x in sub(1, 2)))
print(*(x & 4294967295 for x in sub(-1, -2)))

@micropython.viper
def mul(x: uint, y: uint):
    if False:
        i = 10
        return i + 15
    return (x * y, y * x)
print('mul')
print(*mul(2, 3))
print(*(x & 4294967295 for x in mul(2, -3)))
print(*mul(-2, -3))