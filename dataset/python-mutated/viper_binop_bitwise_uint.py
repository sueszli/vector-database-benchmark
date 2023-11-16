@micropython.viper
def shl(x: uint, y: uint) -> uint:
    if False:
        print('Hello World!')
    return x << y
print('shl')
print(shl(1, 0))
print(shl(1, 30))
print(shl(-1, 10) & 4294967295)

@micropython.viper
def shr(x: uint, y: uint) -> uint:
    if False:
        while True:
            i = 10
    return x >> y
print('shr')
print(shr(1, 0))
print(shr(16, 3))
print(shr(-1, 1) in (2147483647, 9223372036854775807))

@micropython.viper
def and_(x: uint, y: uint):
    if False:
        for i in range(10):
            print('nop')
    return (x & y, y & x)
print('and')
print(*and_(1, 0))
print(*and_(1, 3))
print(*and_(-1, 2))
print(*(x & 4294967295 for x in and_(-1, -2)))

@micropython.viper
def or_(x: uint, y: uint):
    if False:
        for i in range(10):
            print('nop')
    return (x | y, y | x)
print('or')
print(*or_(1, 0))
print(*or_(1, 2))
print(*(x & 4294967295 for x in or_(-1, 2)))

@micropython.viper
def xor(x: uint, y: uint):
    if False:
        return 10
    return (x ^ y, y ^ x)
print('xor')
print(*xor(1, 0))
print(*xor(1, 3))
print(*(x & 4294967295 for x in xor(-1, 3)))
print(*xor(-1, -3))