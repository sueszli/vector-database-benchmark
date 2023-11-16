def setbit(byte, offset, value):
    if False:
        while True:
            i = 10
    '\n    Set a bit in a byte to 1 if value is truthy, 0 if not.\n    '
    if value:
        return byte | 1 << offset
    else:
        return byte & ~(1 << offset)

def getbit(byte, offset):
    if False:
        while True:
            i = 10
    mask = 1 << offset
    return bool(byte & mask)