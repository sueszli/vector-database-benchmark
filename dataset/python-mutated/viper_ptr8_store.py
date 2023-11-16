@micropython.viper
def set(dest: ptr8, val: int):
    if False:
        while True:
            i = 10
    dest[0] = val

@micropython.viper
def set1(dest: ptr8, val: int):
    if False:
        for i in range(10):
            print('nop')
    dest[1] = val

@micropython.viper
def memset(dest: ptr8, val: int, n: int):
    if False:
        print('Hello World!')
    for i in range(n):
        dest[i] = val

@micropython.viper
def memset2(dest_in, val: int):
    if False:
        i = 10
        return i + 15
    dest = ptr8(dest_in)
    n = int(len(dest_in))
    for i in range(n):
        dest[i] = val
b = bytearray(4)
print(b)
set(b, 41)
print(b)
set1(b, 42)
print(b)
memset(b, 43, len(b))
print(b)
memset2(b, 44)
print(b)