import sys
import os
import io
import zlib
import struct
import fnmatch
import pathlib
import urllib.parse
from visidata import vd
__version__ = '0.5.1'

def error(s):
    if False:
        while True:
            i = 10
    raise Exception(s)

def warning(s):
    if False:
        return 10
    print(s, file=sys.stderr)

def get_bits(val: int, *args):
    if False:
        print('Hello World!')
    'Generate bitfields (one for each arg) from LSB to MSB.'
    for n in args:
        x = val & 2 ** n - 1
        val >>= n
        yield x

class RemoteZipInfo:

    def __init__(self, filename: str='', date_time: int=0, header_offset: int=0, compress_type: int=0, compress_size: int=0, file_size: int=0):
        if False:
            return 10
        self.filename = filename
        self.header_offset = header_offset
        self.compress_type = compress_type
        self.compress_size = compress_size
        self.file_size = file_size
        (sec, mins, hour, day, mon, year) = get_bits(date_time, 5, 6, 5, 5, 4, 7)
        self.date_time = (year + 1980, mon, day, hour, mins, sec)

    def is_dir(self):
        if False:
            for i in range(10):
                print('nop')
        return self.filename.endswith('/')

    def parse_extra(self, extra):
        if False:
            return 10
        i = 0
        while i < len(extra):
            (fieldid, fieldsz) = struct.unpack_from('<HH', extra, i)
            i += 4
            if fieldid == 1:
                if fieldsz == 8:
                    fmt = '<Q'
                elif fieldsz == 16:
                    fmt = '<QQ'
                elif fieldsz == 24:
                    fmt = '<QQQ'
                elif fieldsz == 28:
                    fmt = '<QQQI'
                vals = list(struct.unpack_from(fmt, extra, i))
                if self.file_size == 4294967295:
                    self.file_size = vals.pop(0)
                if self.compress_size == 4294967295:
                    self.compress_size = vals.pop(0)
                if self.header_offset == 4294967295:
                    self.header_offset = vals.pop(0)
            i += fieldsz

class RemoteZipFile:
    fmt_eocd = '<IHHHHIIH'
    fmt_eocd64 = '<IQHHIIQQQQ'
    fmt_cdirentry = '<IHHHHIIIIHHHHHII'
    fmt_localhdr = '<IHHHIIIIHH'
    magic_eocd64 = b'PK\x06\x06'
    magic_eocd = b'PK\x05\x06'

    def __init__(self, url):
        if False:
            for i in range(10):
                print('nop')
        urllib3 = vd.importExternal('urllib3')
        self.url = url
        self.http = urllib3.PoolManager()
        self.zip_size = 0

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, a, b, c):
        if False:
            return 10
        pass

    @property
    def files(self):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, '_files'):
            self._files = {r.filename: r for r in self.infoiter()}
        return self._files

    def infolist(self):
        if False:
            print('Hello World!')
        return list(self.infoiter())

    def namelist(self):
        if False:
            print('Hello World!')
        return list((r.filename for r in self.infoiter()))

    def infoiter(self):
        if False:
            return 10
        resp = self.http.request('HEAD', self.url)
        r = resp.headers.get('Accept-Ranges', '')
        if r != 'bytes':
            hostname = urllib.parse.urlparse(self.url).netloc
            warning(f"{hostname} Accept-Ranges header ('{r}') is not 'bytes'--trying anyway")
        self.zip_size = int(resp.headers['Content-Length'])
        resp = self.get_range(self.zip_size - 65536, 65536)
        cdir_start = -1
        i = resp.data.rfind(self.magic_eocd64)
        if i >= 0:
            (magic, eocd_sz, create_ver, min_ver, disk_num, disk_start, disk_num_records, total_num_records, cdir_bytes, cdir_start) = struct.unpack_from(self.fmt_eocd64, resp.data, offset=i)
        else:
            i = resp.data.rfind(self.magic_eocd)
            if i >= 0:
                (magic, disk_num, disk_start, disk_num_records, total_num_records, cdir_bytes, cdir_start, comment_len) = struct.unpack_from(self.fmt_eocd, resp.data, offset=i)
        if cdir_start < 0 or cdir_start >= self.zip_size:
            error('cannot find central directory')
        filehdr_index = len(resp.data) - (self.zip_size - cdir_start)
        if filehdr_index < 0:
            resp = self.get_range(cdir_start, self.zip_size - cdir_start)
            filehdr_index = 0
        cdir_end = filehdr_index + cdir_bytes
        while filehdr_index < cdir_end:
            sizeof_cdirentry = struct.calcsize(self.fmt_cdirentry)
            (magic, ver, ver_needed, flags, method, date_time, crc, complen, uncomplen, fnlen, extralen, commentlen, disknum_start, internal_attr, external_attr, local_header_ofs) = struct.unpack_from(self.fmt_cdirentry, resp.data, offset=filehdr_index)
            filehdr_index += sizeof_cdirentry
            filename = resp.data[filehdr_index:filehdr_index + fnlen]
            filehdr_index += fnlen
            extra = resp.data[filehdr_index:filehdr_index + extralen]
            filehdr_index += extralen
            comment = resp.data[filehdr_index:filehdr_index + commentlen]
            filehdr_index += commentlen
            rzi = RemoteZipInfo(filename.decode(), date_time, local_header_ofs, method, complen, uncomplen)
            rzi.parse_extra(extra)
            yield rzi

    def extract(self, member, path=None, pwd=None):
        if False:
            i = 10
            return i + 15
        if pwd:
            raise NotImplementedError('Passwords not supported yet')
        path = path or pathlib.Path('.')
        outpath = path / member
        os.makedirs(outpath.parent, exist_ok=True)
        with self.open(member) as fpin:
            with open(path / member, mode='wb') as fpout:
                while True:
                    r = fpin.read(65536)
                    if not r:
                        break
                    fpout.write(r)

    def extractall(self, path=None, members=None, pwd=None):
        if False:
            i = 10
            return i + 15
        for fn in members or self.namelist():
            self.extract(fn, path, pwd=pwd)

    def get_range(self, start, n):
        if False:
            while True:
                i = 10
        return self.http.request('GET', self.url, headers={'Range': f'bytes={start}-{start + n - 1}'}, preload_content=False)

    def matching_files(self, *globs):
        if False:
            while True:
                i = 10
        for f in self.files.values():
            if any((fnmatch.fnmatch(f.filename, g) for g in globs)):
                yield f

    def open(self, fn):
        if False:
            i = 10
            return i + 15
        if isinstance(fn, str):
            f = list(self.matching_files(fn))
            if not f:
                error(f'no files matching {fn}')
            f = f[0]
        else:
            f = fn
        sizeof_localhdr = struct.calcsize(self.fmt_localhdr)
        r = self.get_range(f.header_offset, sizeof_localhdr)
        localhdr = struct.unpack_from(self.fmt_localhdr, r.data)
        (magic, ver, flags, method, dos_datetime, _, _, uncomplen, fnlen, extralen) = localhdr
        if method == 0:
            return self.get_range(f.header_offset + sizeof_localhdr + fnlen + extralen, f.compress_size)
        elif method == 8:
            resp = self.get_range(f.header_offset + sizeof_localhdr + fnlen + extralen, f.compress_size)
            return io.BufferedReader(RemoteZipStream(resp, f))
        else:
            error(f'unknown compression method {method}')

    def open(self, fn):
        if False:
            for i in range(10):
                print('nop')
        return io.TextIOWrapper(self.open(fn))

class RemoteZipStream(io.RawIOBase):

    def __init__(self, fp, info):
        if False:
            while True:
                i = 10
        super().__init__()
        self.raw = fp
        self._decompressor = zlib.decompressobj(-15)
        self._buffer = bytes()

    def readable(self):
        if False:
            return 10
        return True

    def readinto(self, b):
        if False:
            print('Hello World!')
        r = self.read(len(b))
        b[:len(r)] = r
        return len(r)

    def read(self, n):
        if False:
            while True:
                i = 10
        while n > len(self._buffer):
            r = self.raw.read(2 ** 18)
            if not r:
                self._buffer += self._decompressor.flush()
                break
            self._buffer += self._decompressor.decompress(r)
        ret = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return ret