@micropython.viper
def get(src: ptr32) -> int:
    if False:
        for i in range(10):
            print('nop')
    return src[0]

@micropython.viper
def get1(src: ptr32) -> int:
    if False:
        return 10
    return src[1]

@micropython.viper
def memadd(src: ptr32, n: int) -> int:
    if False:
        while True:
            i = 10
    sum = 0
    for i in range(n):
        sum += src[i]
    return sum

@micropython.viper
def memadd2(src_in) -> int:
    if False:
        i = 10
        return i + 15
    src = ptr32(src_in)
    n = int(len(src_in)) >> 2
    sum = 0
    for i in range(n):
        sum += src[i]
    return sum
b = bytearray(b'\x12\x12\x12\x124444')
print(b)
print(hex(get(b)), hex(get1(b)))
print(hex(memadd(b, 2)))
print(hex(memadd2(b)))