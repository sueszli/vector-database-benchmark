import snappy
import struct

class HadoopStreamDecompressor(object):
    """This class implements the decompressor-side of the hadoop framing
    format.

    Hadoop fraiming format consists of one or more blocks, each of which is
    composed of one or more compressed subblocks. The block size is the
    uncompressed size, while the subblock size is the size of the compressed
    data.

    https://github.com/andrix/python-snappy/pull/35/files
    """
    __slots__ = ['_buf', '_block_size', '_block_read', '_subblock_size']

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._buf = b''
        self._block_size = None
        self._block_read = 0
        self._subblock_size = None

    def decompress(self, data):
        if False:
            while True:
                i = 10
        self._buf += data
        output = b''
        while True:
            buf = self._decompress_block()
            if len(buf) > 0:
                output += buf
            else:
                break
        return output

    def _decompress_block(self):
        if False:
            while True:
                i = 10
        if self._block_size is None:
            if len(self._buf) <= 4:
                return b''
            self._block_size = struct.unpack('>i', self._buf[:4])[0]
            self._buf = self._buf[4:]
        output = b''
        while self._block_read < self._block_size:
            buf = self._decompress_subblock()
            if len(buf) > 0:
                output += buf
            else:
                break
        if self._block_read == self._block_size:
            self._block_read = 0
            self._block_size = None
        return output

    def _decompress_subblock(self):
        if False:
            while True:
                i = 10
        if self._subblock_size is None:
            if len(self._buf) <= 4:
                return b''
            self._subblock_size = struct.unpack('>i', self._buf[:4])[0]
            self._buf = self._buf[4:]
        if len(self._buf) < self._subblock_size:
            return b''
        compressed = self._buf[:self._subblock_size]
        self._buf = self._buf[self._subblock_size:]
        uncompressed = snappy.uncompress(compressed)
        self._block_read += len(uncompressed)
        self._subblock_size = None
        return uncompressed

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        if self._buf != b'':
            raise snappy.UncompressError('chunk truncated')
        return b''

    def copy(self):
        if False:
            while True:
                i = 10
        copy = HadoopStreamDecompressor()
        copy._buf = self._buf
        copy._block_size = self._block_size
        copy._block_read = self._block_read
        copy._subblock_size = self._subblock_size
        return copy

def hadoop_decompress(src, dst, blocksize=snappy._STREAM_TO_STREAM_BLOCK_SIZE):
    if False:
        return 10
    decompressor = HadoopStreamDecompressor()
    while True:
        buf = src.read(blocksize)
        if not buf:
            break
        buf = decompressor.decompress(buf)
        if buf:
            dst.write(buf)
    decompressor.flush()