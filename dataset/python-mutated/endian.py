import struct

def little_endian_uint32(i):
    if False:
        while True:
            i = 10
    'Return the 32 bit unsigned integer little-endian representation of i'
    s = struct.pack('<I', i)
    return struct.unpack('=I', s)[0]

def big_endian_uint32(i):
    if False:
        for i in range(10):
            print('nop')
    'Return the 32 bit unsigned integer big-endian representation of i'
    s = struct.pack('>I', i)
    return struct.unpack('=I', s)[0]