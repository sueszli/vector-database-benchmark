@micropython.viper
def f1(b: ptr8):
    if False:
        i = 10
        return i + 15
    b[0] += b[1]
b = bytearray(b'\x01\x02')
f1(b)
print(b)

@micropython.viper
def f2(b: ptr8, i: int):
    if False:
        return 10
    b[0] += b[i]
b = bytearray(b'\x01\x02')
f2(b, 1)
print(b)

@micropython.viper
def f3(b: ptr8) -> int:
    if False:
        while True:
            i = 10
    return b[0] << 24 | b[1] << 16 | b[2] << 8 | b[3]
print(hex(f3(b'\x01\x02\x03\x04')))