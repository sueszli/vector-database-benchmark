from collections import deque

def int_to_bytes_big_endian(num):
    if False:
        i = 10
        return i + 15
    bytestr = deque()
    while num > 0:
        bytestr.appendleft(num & 255)
        num >>= 8
    return bytes(bytestr)

def int_to_bytes_little_endian(num):
    if False:
        return 10
    bytestr = []
    while num > 0:
        bytestr.append(num & 255)
        num >>= 8
    return bytes(bytestr)

def bytes_big_endian_to_int(bytestr):
    if False:
        print('Hello World!')
    num = 0
    for b in bytestr:
        num <<= 8
        num += b
    return num

def bytes_little_endian_to_int(bytestr):
    if False:
        return 10
    num = 0
    e = 0
    for b in bytestr:
        num += b << e
        e += 8
    return num