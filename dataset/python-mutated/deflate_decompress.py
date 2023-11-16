try:
    import deflate
    import io
except ImportError:
    print('SKIP')
    raise SystemExit
try:
    empty_gzip = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    deflate.DeflateIO(io.BytesIO(empty_gzip)).read()
except MemoryError:
    print('SKIP')
    raise SystemExit
data_raw = b'\xcb\xcdL.\xca/\xa8,\xc9\xc8\xcfS\xc8H\xcd\xc9\xc9W(\xcf/\xcaIAa\xe7"\xd4\x00\x00'
data_zlib = b'\x18\x95\xcb\xcdL.\xca/\xa8,\xc9\xc8\xcfS\xc8H\xcd\xc9\xc9W(\xcf/\xcaIAa\xe7"\xd4\x00\x00\xbc\xfa\x12\x91'
data_gzip = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03\xcb\xcdL.\xca/\xa8,\xc9\xc8\xcfS\xc8H\xcd\xc9\xc9W(\xcf/\xcaIAa\xe7"\xd4\x00\x00"\xeb\xc4\x98/\x00\x00\x00'
data_wbits_5 = b'\xcbH\xcd\xc9\xc9g\x18\xe9\x00\x08\x88\x95\xcfH\xcd\xc9\xc9\x07\x00'
data_wbits_6 = b'\xcbH\xcd\xc9\xc9g\x18\xe9\x00\x08\x88\xd5\x9f\x91\x9a\x93\x93\x0f\x00'
data_wbits_8 = b'\xcbH\xcd\xc9\xc9g\x18\xe9\x00\x08\x88\xf5\x7fFjNN>\x00'
data_wbits_10 = b"\xcbH\xcd\xc9\xc9g\x18\xe9\x00\x08Fz\x18\x00\xc3`\xa4'\x03`2\x18\xe99\x01\x98\x13Fz\xfe\x07\xe6\xff\x91\x9e\xff\x81\xf9\x7f\xa4\xe7\x7f`\xfe\x1f\xba\xf9?#5''\x1f\x00"

def decompress(data, *args):
    if False:
        while True:
            i = 10
    buf = io.BytesIO(data)
    with deflate.DeflateIO(buf, *args) as g:
        return g.read()

def decompress_error(data, *args):
    if False:
        return 10
    try:
        decompress(data, *args)
    except OSError:
        print('OSError')
    except EOFError:
        print('EOFError')
    except ValueError:
        print('ValueError')
print(decompress(data_raw, deflate.RAW))
print(decompress(data_zlib, deflate.ZLIB))
print(decompress(data_gzip, deflate.GZIP))
print(decompress(data_zlib))
print(decompress(data_gzip))
decompress_error(data_raw)
decompress_error(data_raw, deflate.ZLIB)
decompress_error(data_raw, deflate.GZIP)
decompress_error(data_zlib, deflate.RAW)
decompress_error(data_zlib, deflate.GZIP)
decompress_error(data_gzip, deflate.RAW)
decompress_error(data_gzip, deflate.ZLIB)
decompress_error(b'abcef', deflate.RAW)
decompress_error(b'\x07', deflate.RAW)
decompress_error(data_raw[:10], deflate.RAW)
buf = io.BytesIO(data_zlib)
with deflate.DeflateIO(buf) as g:
    print(buf.seek(0, 1))
    print(g.read(1))
    print(buf.seek(0, 1))
    print(g.read(1))
    print(buf.seek(0, 1))
    print(g.read(2))
    print(buf.seek(0, 1))
    print(g.read())
    print(buf.seek(0, 1))
    print(g.read(1))
    print(buf.seek(0, 1))
    print(g.read())
decompress_error(data_zlib[:-4] + b'\x00\x00\x00\x00')
decompress_error(data_gzip[:-8] + b'\x00\x00\x00\x00\x00\x00\x00\x00')
decompress_error(data_zlib[:-4] + b'\x00\x00\x00\x00', deflate.ZLIB)
decompress_error(data_gzip[:-8] + b'\x00\x00\x00\x00\x00\x00\x00\x00', deflate.GZIP)
b = io.BytesIO(data_raw)
g = deflate.DeflateIO(b, deflate.RAW)
g.read(4)
b.close()
try:
    g.read(4)
except ValueError:
    print('ValueError')
b = io.BytesIO(data_raw)
g = deflate.DeflateIO(b, deflate.RAW)
g.read(4)
g.close()
try:
    g.read(4)
except OSError:
    print('OSError')
data_gzip_header_extra = b'\x1f\x8b\x08\x1e}\x9a\x9bd\x02\x00\x00\x00\x00\x00\x00\xff\xcb\xcdL.\xca/\xa8,\xc9\xc8\xcf\x03\x00\xf2KF>\x0b\x00\x00\x00'
print(decompress(data_gzip_header_extra))
PATTERNS_ZLIB = [(b'0', b'x\x9c3\x00\x00\x001\x001'), (b'a', b'x\x9cK\x04\x00\x00b\x00b'), (b'0' * 100, b'x\x9c30\xa0=\x00\x00\xb3q\x12\xc1'), (bytes(range(64)), b'x\x9cc`dbfaec\xe7\xe0\xe4\xe2\xe6\xe1\xe5\xe3\x17\x10\x14\x12\x16\x11\x15\x13\x97\x90\x94\x92\x96\x91\x95\x93WPTRVQUS\xd7\xd0\xd4\xd2\xd6\xd1\xd5\xd370426153\xb7\xb0\xb4\xb2\xb6\xb1\xb5\xb3\x07\x00\xaa\xe0\x07\xe1'), (b'hello', b'x\x01\x01\x05\x00\xfa\xffhello\x06,\x02\x15'), (b'13371813150|13764518736|12345678901', b'x\x9c\x05\xc1\x81\x01\x000\x04\x04\xb1\x95\\\x1f\xcfn\x86o\x82d\x06Qq\xc8\x9d\xc5X}<e\xb5g\x83\x0f\x89X\x07\xab'), (b'>I}\x00\x951D>I}\x00\x951D>I}\x00\x951D>I}\x00\x951D', b'x\x9c\x05\xc11\x01\x00\x00\x00\x010\x95\x14py\x84\x12C_\x9bR\x8cV\x8a\xd1J1Z)F\x1fw`\x089')]
for (unpacked, packed) in PATTERNS_ZLIB:
    print(decompress(packed) == unpacked)
    print(decompress(packed, deflate.ZLIB) == unpacked)
data_raw_with_footer = data_raw + b'\x00\x00\x00\x00\x00\x00\x00\x00'
print(decompress(data_raw_with_footer, deflate.RAW))
decompress_error(data_wbits_5, deflate.RAW, -1)
print(len(decompress(data_wbits_5, deflate.RAW, 0)))
decompress_error(data_wbits_5, deflate.RAW, 1)
decompress_error(data_wbits_5, deflate.RAW, 4)
for i in range(5, 16):
    print(len(decompress(data_wbits_5, deflate.RAW, i)))
decompress_error(data_wbits_5, deflate.RAW, 16)
decompress_error(data_raw, -1)
decompress_error(data_raw, 5)
decompress_error(data_wbits_6, deflate.RAW, 5)
print(len(decompress(data_wbits_6, deflate.RAW, 6)))
print(len(decompress(data_wbits_6, deflate.RAW, 7)))
decompress_error(data_wbits_8, deflate.RAW, 7)
print(len(decompress(data_wbits_8, deflate.RAW, 8)))
print(len(decompress(data_wbits_8, deflate.RAW, 9)))
decompress_error(data_wbits_10, deflate.RAW)
decompress_error(data_wbits_10, deflate.RAW, 9)
print(len(decompress(data_wbits_10, deflate.RAW, 10)))
data_wbits_10_zlib = b"(\x91\xcbH\xcd\xc9\xc9g\x18\xe9\x00\x08Fz\x18\x00\xc3`\xa4'\x03`2\x18\xe99\x01\x98\x13Fz\xfe\x07\xe6\xff\x91\x9e\xff\x81\xf9\x7f\xa4\xe7\x7f`\xfe\x1f\xba\xf9?#5''\x1f\x00[\xbc\x04)"
print(len(decompress(data_wbits_10_zlib, deflate.ZLIB)))
decompress_error(data_wbits_10_zlib, deflate.ZLIB, 9)
print(len(decompress(data_wbits_10_zlib, deflate.ZLIB, 10)))
print(len(decompress(data_wbits_10_zlib)))