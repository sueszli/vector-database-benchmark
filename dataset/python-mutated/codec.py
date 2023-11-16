from __future__ import absolute_import
import gzip
import io
import platform
import struct
from kafka.vendor import six
from kafka.vendor.six.moves import range
_XERIAL_V1_HEADER = (-126, b'S', b'N', b'A', b'P', b'P', b'Y', 0, 1, 1)
_XERIAL_V1_FORMAT = 'bccccccBii'
ZSTD_MAX_OUTPUT_SIZE = 1024 * 1024
try:
    import snappy
except ImportError:
    snappy = None
try:
    import zstandard as zstd
except ImportError:
    zstd = None
try:
    import lz4.frame as lz4

    def _lz4_compress(payload, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            kwargs.pop('block_linked', None)
            return lz4.compress(payload, block_linked=False, **kwargs)
        except TypeError:
            kwargs.pop('block_mode', None)
            return lz4.compress(payload, block_mode=1, **kwargs)
except ImportError:
    lz4 = None
try:
    import lz4f
except ImportError:
    lz4f = None
try:
    import lz4framed
except ImportError:
    lz4framed = None
try:
    import xxhash
except ImportError:
    xxhash = None
PYPY = bool(platform.python_implementation() == 'PyPy')

def has_gzip():
    if False:
        for i in range(10):
            print('nop')
    return True

def has_snappy():
    if False:
        print('Hello World!')
    return snappy is not None

def has_zstd():
    if False:
        while True:
            i = 10
    return zstd is not None

def has_lz4():
    if False:
        print('Hello World!')
    if lz4 is not None:
        return True
    if lz4f is not None:
        return True
    if lz4framed is not None:
        return True
    return False

def gzip_encode(payload, compresslevel=None):
    if False:
        print('Hello World!')
    if not compresslevel:
        compresslevel = 9
    buf = io.BytesIO()
    gzipper = gzip.GzipFile(fileobj=buf, mode='w', compresslevel=compresslevel)
    try:
        gzipper.write(payload)
    finally:
        gzipper.close()
    return buf.getvalue()

def gzip_decode(payload):
    if False:
        return 10
    buf = io.BytesIO(payload)
    gzipper = gzip.GzipFile(fileobj=buf, mode='r')
    try:
        return gzipper.read()
    finally:
        gzipper.close()

def snappy_encode(payload, xerial_compatible=True, xerial_blocksize=32 * 1024):
    if False:
        i = 10
        return i + 15
    'Encodes the given data with snappy compression.\n\n    If xerial_compatible is set then the stream is encoded in a fashion\n    compatible with the xerial snappy library.\n\n    The block size (xerial_blocksize) controls how frequent the blocking occurs\n    32k is the default in the xerial library.\n\n    The format winds up being:\n\n\n        +-------------+------------+--------------+------------+--------------+\n        |   Header    | Block1 len | Block1 data  | Blockn len | Blockn data  |\n        +-------------+------------+--------------+------------+--------------+\n        |  16 bytes   |  BE int32  | snappy bytes |  BE int32  | snappy bytes |\n        +-------------+------------+--------------+------------+--------------+\n\n\n    It is important to note that the blocksize is the amount of uncompressed\n    data presented to snappy at each block, whereas the blocklen is the number\n    of bytes that will be present in the stream; so the length will always be\n    <= blocksize.\n\n    '
    if not has_snappy():
        raise NotImplementedError('Snappy codec is not available')
    if not xerial_compatible:
        return snappy.compress(payload)
    out = io.BytesIO()
    for (fmt, dat) in zip(_XERIAL_V1_FORMAT, _XERIAL_V1_HEADER):
        out.write(struct.pack('!' + fmt, dat))
    if PYPY:
        chunker = lambda payload, i, size: payload[i:size + i]
    elif six.PY2:
        chunker = lambda payload, i, size: buffer(payload, i, size)
    else:
        chunker = lambda payload, i, size: memoryview(payload)[i:size + i].tobytes()
    for chunk in (chunker(payload, i, xerial_blocksize) for i in range(0, len(payload), xerial_blocksize)):
        block = snappy.compress(chunk)
        block_size = len(block)
        out.write(struct.pack('!i', block_size))
        out.write(block)
    return out.getvalue()

def _detect_xerial_stream(payload):
    if False:
        return 10
    "Detects if the data given might have been encoded with the blocking mode\n        of the xerial snappy library.\n\n        This mode writes a magic header of the format:\n            +--------+--------------+------------+---------+--------+\n            | Marker | Magic String | Null / Pad | Version | Compat |\n            +--------+--------------+------------+---------+--------+\n            |  byte  |   c-string   |    byte    |  int32  | int32  |\n            +--------+--------------+------------+---------+--------+\n            |  -126  |   'SNAPPY'   |     \x00     |         |        |\n            +--------+--------------+------------+---------+--------+\n\n        The pad appears to be to ensure that SNAPPY is a valid cstring\n        The version is the version of this format as written by xerial,\n        in the wild this is currently 1 as such we only support v1.\n\n        Compat is there to claim the minimum supported version that\n        can read a xerial block stream, presently in the wild this is\n        1.\n    "
    if len(payload) > 16:
        header = struct.unpack('!' + _XERIAL_V1_FORMAT, bytes(payload)[:16])
        return header == _XERIAL_V1_HEADER
    return False

def snappy_decode(payload):
    if False:
        print('Hello World!')
    if not has_snappy():
        raise NotImplementedError('Snappy codec is not available')
    if _detect_xerial_stream(payload):
        out = io.BytesIO()
        byt = payload[16:]
        length = len(byt)
        cursor = 0
        while cursor < length:
            block_size = struct.unpack_from('!i', byt[cursor:])[0]
            cursor += 4
            end = cursor + block_size
            out.write(snappy.decompress(byt[cursor:end]))
            cursor = end
        out.seek(0)
        return out.read()
    else:
        return snappy.decompress(payload)
if lz4:
    lz4_encode = _lz4_compress
elif lz4f:
    lz4_encode = lz4f.compressFrame
elif lz4framed:
    lz4_encode = lz4framed.compress
else:
    lz4_encode = None

def lz4f_decode(payload):
    if False:
        for i in range(10):
            print('nop')
    'Decode payload using interoperable LZ4 framing. Requires Kafka >= 0.10'
    ctx = lz4f.createDecompContext()
    data = lz4f.decompressFrame(payload, ctx)
    lz4f.freeDecompContext(ctx)
    if data['next'] != 0:
        raise RuntimeError('lz4f unable to decompress full payload')
    return data['decomp']
if lz4:
    lz4_decode = lz4.decompress
elif lz4f:
    lz4_decode = lz4f_decode
elif lz4framed:
    lz4_decode = lz4framed.decompress
else:
    lz4_decode = None

def lz4_encode_old_kafka(payload):
    if False:
        i = 10
        return i + 15
    'Encode payload for 0.8/0.9 brokers -- requires an incorrect header checksum.'
    assert xxhash is not None
    data = lz4_encode(payload)
    header_size = 7
    flg = data[4]
    if not isinstance(flg, int):
        flg = ord(flg)
    content_size_bit = flg >> 3 & 1
    if content_size_bit:
        flg -= 8
        data = bytearray(data)
        data[4] = flg
        data = bytes(data)
        payload = data[header_size + 8:]
    else:
        payload = data[header_size:]
    hc = xxhash.xxh32(data[0:header_size - 1]).digest()[-2:-1]
    return b''.join([data[0:header_size - 1], hc, payload])

def lz4_decode_old_kafka(payload):
    if False:
        return 10
    assert xxhash is not None
    header_size = 7
    if isinstance(payload[4], int):
        flg = payload[4]
    else:
        flg = ord(payload[4])
    content_size_bit = flg >> 3 & 1
    if content_size_bit:
        header_size += 8
    hc = xxhash.xxh32(payload[4:header_size - 1]).digest()[-2:-1]
    munged_payload = b''.join([payload[0:header_size - 1], hc, payload[header_size:]])
    return lz4_decode(munged_payload)

def zstd_encode(payload):
    if False:
        print('Hello World!')
    if not zstd:
        raise NotImplementedError('Zstd codec is not available')
    return zstd.ZstdCompressor().compress(payload)

def zstd_decode(payload):
    if False:
        i = 10
        return i + 15
    if not zstd:
        raise NotImplementedError('Zstd codec is not available')
    try:
        return zstd.ZstdDecompressor().decompress(payload)
    except zstd.ZstdError:
        return zstd.ZstdDecompressor().decompress(payload, max_output_size=ZSTD_MAX_OUTPUT_SIZE)