@micropython.viper
def get_addr(x: ptr) -> ptr:
    if False:
        print('Hello World!')
    return x

@micropython.viper
def memset(dest: ptr8, c: int, n: int):
    if False:
        i = 10
        return i + 15
    for i in range(n):
        dest[i] = c

@micropython.viper
def memsum(src: ptr8, n: int) -> int:
    if False:
        i = 10
        return i + 15
    s = 0
    for i in range(n):
        s += src[i]
    return s
ar = bytearray(b'0000')
addr = get_addr(ar)
print(type(ar))
print(type(addr))
print(ar)
memset(ar, ord('1'), len(ar))
print(ar)
memset(addr, ord('2'), len(ar))
print(ar)
memset(addr + 2, ord('3'), len(ar) - 2)
print(ar)
print(memsum(b'\x01\x02\x03\x04', 4))