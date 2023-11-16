__license__ = 'GPL v3'
__copyright__ = '2013, Kovid Goyal <kovid at kovidgoyal.net>'
import sys
from struct import unpack_from
from collections import namedtuple
from calibre.utils.wmf import create_bmp_from_dib, to_png
from polyglot.builtins import iteritems
RECORD_TYPES = {'EMR_BITBLT': 76, 'EMR_STRETCHBLT': 77, 'EMR_MASKBLT': 78, 'EMR_PLGBLT': 79, 'EMR_SETDIBITSTODEVICE': 80, 'EMR_STRETCHDIBITS': 81, 'EMR_ALPHABLEND': 114, 'EMR_TRANSPARENTBLT': 116, 'EOF': 14, 'HEADER': 1}
RECORD_RMAP = {v: k for (k, v) in iteritems(RECORD_TYPES)}
StretchDiBits = namedtuple('StretchDiBits', 'left top right bottom x_dest y_dest x_src y_src cx_src cy_src bmp_hdr_offset bmp_header_size bmp_bits_offset bmp_bits_size usage op dest_width dest_height')

class EMF:

    def __init__(self, raw, verbose=0):
        if False:
            return 10
        self.pos = 0
        self.found_eof = False
        self.verbose = verbose
        self.func_map = {v: getattr(self, 'handle_%s' % k.replace('EMR_', '').lower(), self.handle_unknown) for (k, v) in iteritems(RECORD_TYPES)}
        self.bitmaps = []
        while self.pos < len(raw) and (not self.found_eof):
            self.read_record(raw)
        self.has_raster_image = bool(self.bitmaps)

    def handle_unknown(self, rtype, size, raw):
        if False:
            i = 10
            return i + 15
        if self.verbose:
            print('Ignoring unknown record:', RECORD_RMAP.get(rtype, hex(rtype).upper()))

    def handle_header(self, rtype, size, raw):
        if False:
            while True:
                i = 10
        pass

    def handle_stretchdibits(self, rtype, size, raw):
        if False:
            for i in range(10):
                print('nop')
        data = StretchDiBits(*unpack_from(b'<18I', raw, 8))
        hdr = raw[data.bmp_hdr_offset:data.bmp_hdr_offset + data.bmp_header_size]
        bits = raw[data.bmp_bits_offset:data.bmp_bits_offset + data.bmp_bits_size]
        bmp = create_bmp_from_dib(hdr + bits)
        self.bitmaps.append(bmp)

    def handle_eof(self, rtype, size, raw):
        if False:
            for i in range(10):
                print('nop')
        self.found_eof = True

    def read_record(self, raw):
        if False:
            while True:
                i = 10
        (rtype, size) = unpack_from(b'<II', raw, self.pos)
        record = raw[self.pos:self.pos + size]
        self.pos += size
        self.func_map.get(rtype, self.handle_unknown)(rtype, size, record)

    def to_png(self):
        if False:
            print('Hello World!')
        bmps = list(sorted(self.bitmaps, key=lambda x: len(x)))
        bmp = bmps[-1]
        return to_png(bmp)

def emf_unwrap(raw, verbose=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the largest embedded raster image in the EMF.\n    The returned data is in PNG format.\n    '
    w = EMF(raw, verbose=verbose)
    if not w.has_raster_image:
        raise ValueError('No raster image found in the EMF')
    return w.to_png()
if __name__ == '__main__':
    with open(sys.argv[-1], 'rb') as f:
        raw = f.read()
    emf = EMF(raw, verbose=4)
    open('/t/test.bmp', 'wb').write(emf.bitmaps[0])
    open('/t/test.png', 'wb').write(emf.to_png())