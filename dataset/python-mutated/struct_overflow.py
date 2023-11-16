import skip_if
skip_if.no_bigint()
try:
    import struct
except ImportError:
    print('SKIP')
    raise SystemExit

def test_struct_overflow(typecode, val):
    if False:
        return 10
    try:
        print(struct.pack(typecode, val))
    except OverflowError:
        print('OverflowError')
    except struct.error:
        print('OverflowError')
test_struct_overflow('>Q', -1)
test_struct_overflow('>L', -1)
test_struct_overflow('>I', -1)
test_struct_overflow('>H', -1)
test_struct_overflow('>B', -1)
test_struct_overflow('>Q', -2 ** 64 // 2 ** 64)
test_struct_overflow('>L', -2 ** 64 // 2 ** 64)
test_struct_overflow('>I', -2 ** 64 // 2 ** 64)
test_struct_overflow('>H', -2 ** 64 // 2 ** 64)
test_struct_overflow('>B', -2 ** 64 // 2 ** 64)
test_struct_overflow('>q', 2 ** 63)
test_struct_overflow('>l', 2 ** 31)
test_struct_overflow('>i', 2 ** 31)
test_struct_overflow('>h', 2 ** 15)
test_struct_overflow('>b', 2 ** 7)
test_struct_overflow('>q', 2 ** 64 // 2 ** 1)
test_struct_overflow('>l', 2 ** 64 // 2 ** 33)
test_struct_overflow('>i', 2 ** 64 // 2 ** 33)
test_struct_overflow('>h', 2 ** 64 // 2 ** 49)
test_struct_overflow('>b', 2 ** 64 // 2 ** 57)