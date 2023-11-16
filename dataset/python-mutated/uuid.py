import re
from random import randrange
from struct import pack, unpack
try:
    long
except NameError:
    long = int

def generate():
    if False:
        while True:
            i = 10
    top = (1 << 31) - 1
    return pack('IIII', randrange(top), randrange(top), randrange(top), randrange(top))

def bin_to_string(uuid):
    if False:
        for i in range(10):
            print('nop')
    (uuid1, uuid2, uuid3) = unpack('<LHH', uuid[:8])
    (uuid4, uuid5, uuid6) = unpack('>HHL', uuid[8:16])
    return '%08X-%04X-%04X-%04X-%04X%08X' % (uuid1, uuid2, uuid3, uuid4, uuid5, uuid6)

def string_to_bin(uuid):
    if False:
        print('Hello World!')
    matches = re.match('([\\dA-Fa-f]{8})-([\\dA-Fa-f]{4})-([\\dA-Fa-f]{4})-([\\dA-Fa-f]{4})-([\\dA-Fa-f]{4})([\\dA-Fa-f]{8})', uuid)
    (uuid1, uuid2, uuid3, uuid4, uuid5, uuid6) = map(lambda x: long(x, 16), matches.groups())
    uuid = pack('<LHH', uuid1, uuid2, uuid3)
    uuid += pack('>HHL', uuid4, uuid5, uuid6)
    return uuid

def stringver_to_bin(s):
    if False:
        i = 10
        return i + 15
    (maj, min) = s.split('.')
    return pack('<H', int(maj)) + pack('<H', int(min))

def uuidtup_to_bin(tup):
    if False:
        i = 10
        return i + 15
    if len(tup) != 2:
        return
    return string_to_bin(tup[0]) + stringver_to_bin(tup[1])

def bin_to_uuidtup(bin):
    if False:
        i = 10
        return i + 15
    assert len(bin) == 20
    uuidstr = bin_to_string(bin[:16])
    (maj, min) = unpack('<HH', bin[16:])
    return (uuidstr, '%d.%d' % (maj, min))

def string_to_uuidtup(s):
    if False:
        i = 10
        return i + 15
    g = re.search('([A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{12}).*?([0-9]{1,5}\\.[0-9]{1,5})', s + ' 1.0')
    if g:
        (u, v) = g.groups()
        return (u, v)
    return

def uuidtup_to_string(tup):
    if False:
        while True:
            i = 10
    (uuid, (maj, min)) = tup
    return '%s v%d.%d' % (uuid, maj, min)