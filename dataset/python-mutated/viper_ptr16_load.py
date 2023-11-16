@micropython.viper
def get(src: ptr16) -> int:
    if False:
        print('Hello World!')
    return src[0]

@micropython.viper
def get1(src: ptr16) -> int:
    if False:
        print('Hello World!')
    return src[1]

@micropython.viper
def memadd(src: ptr16, n: int) -> int:
    if False:
        i = 10
        return i + 15
    sum = 0
    for i in range(n):
        sum += src[i]
    return sum

@micropython.viper
def memadd2(src_in) -> int:
    if False:
        print('Hello World!')
    src = ptr16(src_in)
    n = int(len(src_in)) >> 1
    sum = 0
    for i in range(n):
        sum += src[i]
    return sum
b = bytearray(b'1234')
print(b)
print(get(b), get1(b))
print(memadd(b, 2))
print(memadd2(b))