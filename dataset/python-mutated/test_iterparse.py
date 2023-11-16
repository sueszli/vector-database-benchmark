import io
import zlib
from astropy.utils.xml.iterparser import _fast_iterparse
HEADER = '<?xml version="1.0" encoding="UTF-8"?>\n<VOTABLE>\n <RESOURCE type="results">\n  <TABLE>\n   <FIELD ID="foo" name="foo" datatype="int" arraysize="1"/>\n    <DATA>\n     <TABLEDATA>\n'
ROW = '<TR><TD>0</TD></TR>\n'
FOOTER = '\n    </TABLEDATA>\n   </DATA>\n  </TABLE>\n </RESOURCE>\n</VOTABLE>\n'
VOTABLE_XML = HEADER + 125 * ROW + FOOTER

class UngzipFileWrapper:

    def __init__(self, fd, **kwargs):
        if False:
            while True:
                i = 10
        self._file = fd
        self._z = zlib.decompressobj(16 + zlib.MAX_WBITS)

    def read(self, requested_length):
        if False:
            return 10
        clamped_length = max(1, min(1 << 24, requested_length))
        compressed = self._file.read(clamped_length)
        plaintext = self._z.decompress(compressed)
        if len(compressed) == 0:
            self.close()
        return plaintext

    def __getattr__(self, attr):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._file, attr)

def test_iterparser_over_read_simple():
    if False:
        while True:
            i = 10
    zlib_GZIP_STYLE_HEADER = 16
    compo = zlib.compressobj(zlib.Z_BEST_COMPRESSION, zlib.DEFLATED, zlib.MAX_WBITS + zlib_GZIP_STYLE_HEADER)
    s = compo.compress(VOTABLE_XML.encode())
    s = s + compo.flush()
    fd = io.BytesIO(s)
    fd.seek(0)
    MINIMUM_REQUESTABLE_BUFFER_SIZE = 1024
    uncompressed_fd = UngzipFileWrapper(fd)
    iterable = _fast_iterparse(uncompressed_fd.read, MINIMUM_REQUESTABLE_BUFFER_SIZE)
    list(iterable)