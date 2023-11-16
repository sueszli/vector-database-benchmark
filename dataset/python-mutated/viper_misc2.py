@micropython.viper
def expand(dest: ptr8, source: ptr8, length: int):
    if False:
        print('Hello World!')
    n = 0
    for x in range(0, length, 2):
        c = source[x]
        d = source[x + 1]
        dest[n] = c & 224 | (c & 28) >> 1
        n += 1
        dest[n] = (c & 3) << 6 | (d & 224) >> 4
        n += 1
        dest[n] = (d & 28) << 3 | (d & 3) << 2
        n += 1
source = b'\xaa\xaa\xff\xff'
dest = bytearray(len(source) // 2 * 3)
expand(dest, source, len(source))
print(dest)