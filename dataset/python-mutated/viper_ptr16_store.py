@micropython.viper
def set(dest: ptr16, val: int):
    if False:
        print('Hello World!')
    dest[0] = val

@micropython.viper
def set1(dest: ptr16, val: int):
    if False:
        while True:
            i = 10
    dest[1] = val

@micropython.viper
def memset(dest: ptr16, val: int, n: int):
    if False:
        i = 10
        return i + 15
    for i in range(n):
        dest[i] = val

@micropython.viper
def memset2(dest_in, val: int):
    if False:
        for i in range(10):
            print('nop')
    dest = ptr16(dest_in)
    n = int(len(dest_in)) >> 1
    for i in range(n):
        dest[i] = val
b = bytearray(4)
print(b)
set(b, 16962)
print(b)
set1(b, 17219)
print(b)
memset(b, 17476, len(b) // 2)
print(b)
memset2(b, 17733)
print(b)