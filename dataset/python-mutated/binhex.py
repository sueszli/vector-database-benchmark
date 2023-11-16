"""Macintosh binhex compression/decompression.

easy interface:
binhex(inputfilename, outputfilename)
hexbin(inputfilename, outputfilename)
"""
import binascii
import contextlib
import io
import os
import struct
import warnings
warnings.warn('the binhex module is deprecated', DeprecationWarning, stacklevel=2)
__all__ = ['binhex', 'hexbin', 'Error']

class Error(Exception):
    pass
_DID_HEADER = 0
_DID_DATA = 1
REASONABLY_LARGE = 32768
LINELEN = 64
RUNCHAR = b'\x90'

class FInfo:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.Type = '????'
        self.Creator = '????'
        self.Flags = 0

def getfileinfo(name):
    if False:
        i = 10
        return i + 15
    finfo = FInfo()
    with io.open(name, 'rb') as fp:
        data = fp.read(512)
        if 0 not in data:
            finfo.Type = 'TEXT'
        fp.seek(0, 2)
        dsize = fp.tell()
    (dir, file) = os.path.split(name)
    file = file.replace(':', '-', 1)
    return (file, finfo, dsize, 0)

class openrsrc:

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        pass

    def read(self, *args):
        if False:
            while True:
                i = 10
        return b''

    def write(self, *args):
        if False:
            while True:
                i = 10
        pass

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        pass

@contextlib.contextmanager
def _ignore_deprecation_warning():
    if False:
        return 10
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '', DeprecationWarning)
        yield

class _Hqxcoderengine:
    """Write data to the coder in 3-byte chunks"""

    def __init__(self, ofp):
        if False:
            while True:
                i = 10
        self.ofp = ofp
        self.data = b''
        self.hqxdata = b''
        self.linelen = LINELEN - 1

    def write(self, data):
        if False:
            i = 10
            return i + 15
        self.data = self.data + data
        datalen = len(self.data)
        todo = datalen // 3 * 3
        data = self.data[:todo]
        self.data = self.data[todo:]
        if not data:
            return
        with _ignore_deprecation_warning():
            self.hqxdata = self.hqxdata + binascii.b2a_hqx(data)
        self._flush(0)

    def _flush(self, force):
        if False:
            i = 10
            return i + 15
        first = 0
        while first <= len(self.hqxdata) - self.linelen:
            last = first + self.linelen
            self.ofp.write(self.hqxdata[first:last] + b'\r')
            self.linelen = LINELEN
            first = last
        self.hqxdata = self.hqxdata[first:]
        if force:
            self.ofp.write(self.hqxdata + b':\r')

    def close(self):
        if False:
            while True:
                i = 10
        if self.data:
            with _ignore_deprecation_warning():
                self.hqxdata = self.hqxdata + binascii.b2a_hqx(self.data)
        self._flush(1)
        self.ofp.close()
        del self.ofp

class _Rlecoderengine:
    """Write data to the RLE-coder in suitably large chunks"""

    def __init__(self, ofp):
        if False:
            i = 10
            return i + 15
        self.ofp = ofp
        self.data = b''

    def write(self, data):
        if False:
            print('Hello World!')
        self.data = self.data + data
        if len(self.data) < REASONABLY_LARGE:
            return
        with _ignore_deprecation_warning():
            rledata = binascii.rlecode_hqx(self.data)
        self.ofp.write(rledata)
        self.data = b''

    def close(self):
        if False:
            i = 10
            return i + 15
        if self.data:
            with _ignore_deprecation_warning():
                rledata = binascii.rlecode_hqx(self.data)
            self.ofp.write(rledata)
        self.ofp.close()
        del self.ofp

class BinHex:

    def __init__(self, name_finfo_dlen_rlen, ofp):
        if False:
            return 10
        (name, finfo, dlen, rlen) = name_finfo_dlen_rlen
        close_on_error = False
        if isinstance(ofp, str):
            ofname = ofp
            ofp = io.open(ofname, 'wb')
            close_on_error = True
        try:
            ofp.write(b'(This file must be converted with BinHex 4.0)\r\r:')
            hqxer = _Hqxcoderengine(ofp)
            self.ofp = _Rlecoderengine(hqxer)
            self.crc = 0
            if finfo is None:
                finfo = FInfo()
            self.dlen = dlen
            self.rlen = rlen
            self._writeinfo(name, finfo)
            self.state = _DID_HEADER
        except:
            if close_on_error:
                ofp.close()
            raise

    def _writeinfo(self, name, finfo):
        if False:
            for i in range(10):
                print('nop')
        nl = len(name)
        if nl > 63:
            raise Error('Filename too long')
        d = bytes([nl]) + name.encode('latin-1') + b'\x00'
        (tp, cr) = (finfo.Type, finfo.Creator)
        if isinstance(tp, str):
            tp = tp.encode('latin-1')
        if isinstance(cr, str):
            cr = cr.encode('latin-1')
        d2 = tp + cr
        d3 = struct.pack('>h', finfo.Flags)
        d4 = struct.pack('>ii', self.dlen, self.rlen)
        info = d + d2 + d3 + d4
        self._write(info)
        self._writecrc()

    def _write(self, data):
        if False:
            return 10
        self.crc = binascii.crc_hqx(data, self.crc)
        self.ofp.write(data)

    def _writecrc(self):
        if False:
            while True:
                i = 10
        if self.crc < 0:
            fmt = '>h'
        else:
            fmt = '>H'
        self.ofp.write(struct.pack(fmt, self.crc))
        self.crc = 0

    def write(self, data):
        if False:
            while True:
                i = 10
        if self.state != _DID_HEADER:
            raise Error('Writing data at the wrong time')
        self.dlen = self.dlen - len(data)
        self._write(data)

    def close_data(self):
        if False:
            return 10
        if self.dlen != 0:
            raise Error('Incorrect data size, diff=%r' % (self.rlen,))
        self._writecrc()
        self.state = _DID_DATA

    def write_rsrc(self, data):
        if False:
            return 10
        if self.state < _DID_DATA:
            self.close_data()
        if self.state != _DID_DATA:
            raise Error('Writing resource data at the wrong time')
        self.rlen = self.rlen - len(data)
        self._write(data)

    def close(self):
        if False:
            print('Hello World!')
        if self.state is None:
            return
        try:
            if self.state < _DID_DATA:
                self.close_data()
            if self.state != _DID_DATA:
                raise Error('Close at the wrong time')
            if self.rlen != 0:
                raise Error('Incorrect resource-datasize, diff=%r' % (self.rlen,))
            self._writecrc()
        finally:
            self.state = None
            ofp = self.ofp
            del self.ofp
            ofp.close()

def binhex(inp, out):
    if False:
        for i in range(10):
            print('nop')
    'binhex(infilename, outfilename): create binhex-encoded copy of a file'
    finfo = getfileinfo(inp)
    ofp = BinHex(finfo, out)
    with io.open(inp, 'rb') as ifp:
        while True:
            d = ifp.read(128000)
            if not d:
                break
            ofp.write(d)
        ofp.close_data()
    ifp = openrsrc(inp, 'rb')
    while True:
        d = ifp.read(128000)
        if not d:
            break
        ofp.write_rsrc(d)
    ofp.close()
    ifp.close()

class _Hqxdecoderengine:
    """Read data via the decoder in 4-byte chunks"""

    def __init__(self, ifp):
        if False:
            return 10
        self.ifp = ifp
        self.eof = 0

    def read(self, totalwtd):
        if False:
            print('Hello World!')
        'Read at least wtd bytes (or until EOF)'
        decdata = b''
        wtd = totalwtd
        while wtd > 0:
            if self.eof:
                return decdata
            wtd = (wtd + 2) // 3 * 4
            data = self.ifp.read(wtd)
            while True:
                try:
                    with _ignore_deprecation_warning():
                        (decdatacur, self.eof) = binascii.a2b_hqx(data)
                    break
                except binascii.Incomplete:
                    pass
                newdata = self.ifp.read(1)
                if not newdata:
                    raise Error('Premature EOF on binhex file')
                data = data + newdata
            decdata = decdata + decdatacur
            wtd = totalwtd - len(decdata)
            if not decdata and (not self.eof):
                raise Error('Premature EOF on binhex file')
        return decdata

    def close(self):
        if False:
            while True:
                i = 10
        self.ifp.close()

class _Rledecoderengine:
    """Read data via the RLE-coder"""

    def __init__(self, ifp):
        if False:
            print('Hello World!')
        self.ifp = ifp
        self.pre_buffer = b''
        self.post_buffer = b''
        self.eof = 0

    def read(self, wtd):
        if False:
            print('Hello World!')
        if wtd > len(self.post_buffer):
            self._fill(wtd - len(self.post_buffer))
        rv = self.post_buffer[:wtd]
        self.post_buffer = self.post_buffer[wtd:]
        return rv

    def _fill(self, wtd):
        if False:
            while True:
                i = 10
        self.pre_buffer = self.pre_buffer + self.ifp.read(wtd + 4)
        if self.ifp.eof:
            with _ignore_deprecation_warning():
                self.post_buffer = self.post_buffer + binascii.rledecode_hqx(self.pre_buffer)
            self.pre_buffer = b''
            return
        mark = len(self.pre_buffer)
        if self.pre_buffer[-3:] == RUNCHAR + b'\x00' + RUNCHAR:
            mark = mark - 3
        elif self.pre_buffer[-1:] == RUNCHAR:
            mark = mark - 2
        elif self.pre_buffer[-2:] == RUNCHAR + b'\x00':
            mark = mark - 2
        elif self.pre_buffer[-2:-1] == RUNCHAR:
            pass
        else:
            mark = mark - 1
        with _ignore_deprecation_warning():
            self.post_buffer = self.post_buffer + binascii.rledecode_hqx(self.pre_buffer[:mark])
        self.pre_buffer = self.pre_buffer[mark:]

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.ifp.close()

class HexBin:

    def __init__(self, ifp):
        if False:
            return 10
        if isinstance(ifp, str):
            ifp = io.open(ifp, 'rb')
        while True:
            ch = ifp.read(1)
            if not ch:
                raise Error('No binhex data found')
            if ch == b'\r':
                continue
            if ch == b':':
                break
        hqxifp = _Hqxdecoderengine(ifp)
        self.ifp = _Rledecoderengine(hqxifp)
        self.crc = 0
        self._readheader()

    def _read(self, len):
        if False:
            while True:
                i = 10
        data = self.ifp.read(len)
        self.crc = binascii.crc_hqx(data, self.crc)
        return data

    def _checkcrc(self):
        if False:
            while True:
                i = 10
        filecrc = struct.unpack('>h', self.ifp.read(2))[0] & 65535
        self.crc = self.crc & 65535
        if filecrc != self.crc:
            raise Error('CRC error, computed %x, read %x' % (self.crc, filecrc))
        self.crc = 0

    def _readheader(self):
        if False:
            while True:
                i = 10
        len = self._read(1)
        fname = self._read(ord(len))
        rest = self._read(1 + 4 + 4 + 2 + 4 + 4)
        self._checkcrc()
        type = rest[1:5]
        creator = rest[5:9]
        flags = struct.unpack('>h', rest[9:11])[0]
        self.dlen = struct.unpack('>l', rest[11:15])[0]
        self.rlen = struct.unpack('>l', rest[15:19])[0]
        self.FName = fname
        self.FInfo = FInfo()
        self.FInfo.Creator = creator
        self.FInfo.Type = type
        self.FInfo.Flags = flags
        self.state = _DID_HEADER

    def read(self, *n):
        if False:
            i = 10
            return i + 15
        if self.state != _DID_HEADER:
            raise Error('Read data at wrong time')
        if n:
            n = n[0]
            n = min(n, self.dlen)
        else:
            n = self.dlen
        rv = b''
        while len(rv) < n:
            rv = rv + self._read(n - len(rv))
        self.dlen = self.dlen - n
        return rv

    def close_data(self):
        if False:
            for i in range(10):
                print('nop')
        if self.state != _DID_HEADER:
            raise Error('close_data at wrong time')
        if self.dlen:
            dummy = self._read(self.dlen)
        self._checkcrc()
        self.state = _DID_DATA

    def read_rsrc(self, *n):
        if False:
            i = 10
            return i + 15
        if self.state == _DID_HEADER:
            self.close_data()
        if self.state != _DID_DATA:
            raise Error('Read resource data at wrong time')
        if n:
            n = n[0]
            n = min(n, self.rlen)
        else:
            n = self.rlen
        self.rlen = self.rlen - n
        return self._read(n)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        if self.state is None:
            return
        try:
            if self.rlen:
                dummy = self.read_rsrc(self.rlen)
            self._checkcrc()
        finally:
            self.state = None
            self.ifp.close()

def hexbin(inp, out):
    if False:
        return 10
    'hexbin(infilename, outfilename) - Decode binhexed file'
    ifp = HexBin(inp)
    finfo = ifp.FInfo
    if not out:
        out = ifp.FName
    with io.open(out, 'wb') as ofp:
        while True:
            d = ifp.read(128000)
            if not d:
                break
            ofp.write(d)
    ifp.close_data()
    d = ifp.read_rsrc(128000)
    if d:
        ofp = openrsrc(out, 'wb')
        ofp.write(d)
        while True:
            d = ifp.read_rsrc(128000)
            if not d:
                break
            ofp.write(d)
        ofp.close()
    ifp.close()