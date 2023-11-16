@micropython.viper
def set(dest: ptr32, val: int):
    if False:
        for i in range(10):
            print('nop')
    dest[0] = val

@micropython.viper
def set1(dest: ptr32, val: int):
    if False:
        print('Hello World!')
    dest[1] = val

@micropython.viper
def memset(dest: ptr32, val: int, n: int):
    if False:
        i = 10
        return i + 15
    for i in range(n):
        dest[i] = val

@micropython.viper
def memset2(dest_in, val: int):
    if False:
        while True:
            i = 10
    dest = ptr32(dest_in)
    n = int(len(dest_in)) >> 2
    for i in range(n):
        dest[i] = val
b = bytearray(8)
print(b)
set(b, 1111638594)
print(b)
set1(b, 1128481603)
print(b)
memset(b, 1145324612, len(b) // 4)
print(b)
memset2(b, 1162167621)
print(b)