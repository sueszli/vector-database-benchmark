from abc import ABC
from typing import Final
from base64 import b64decode
from io import BytesIO
import struct
from redbot import VersionInfo
from redbot.core import commands
from ..converters import get_lazy_converter, get_playlist_converter
__version__ = VersionInfo.from_json({'major': 2, 'minor': 5, 'micro': 0, 'releaselevel': 'final'})
__author__ = ['aikaterna', 'Draper']
_SCHEMA_VERSION: Final[int] = 3
_OWNER_NOTIFICATION: Final[int] = 1
LazyGreedyConverter = get_lazy_converter('--')
PlaylistConverter = get_playlist_converter()

class CompositeMetaClass(type(commands.Cog), type(ABC)):
    """
    This allows the metaclass used for proper type detection to
    coexist with discord.py's metaclass
    """
    pass

class DataReader:

    def __init__(self, ts):
        if False:
            while True:
                i = 10
        self._buf = BytesIO(b64decode(ts))

    def _read(self, n):
        if False:
            i = 10
            return i + 15
        return self._buf.read(n)

    def read_byte(self):
        if False:
            i = 10
            return i + 15
        return self._read(1)

    def read_boolean(self):
        if False:
            for i in range(10):
                print('nop')
        (result,) = struct.unpack('B', self.read_byte())
        return result != 0

    def read_unsigned_short(self):
        if False:
            print('Hello World!')
        (result,) = struct.unpack('>H', self._read(2))
        return result

    def read_int(self):
        if False:
            for i in range(10):
                print('nop')
        (result,) = struct.unpack('>i', self._read(4))
        return result

    def read_long(self):
        if False:
            i = 10
            return i + 15
        (result,) = struct.unpack('>Q', self._read(8))
        return result

    def read_utf(self):
        if False:
            for i in range(10):
                print('nop')
        text_length = self.read_unsigned_short()
        return self._read(text_length)

class DataWriter:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._buf = BytesIO()

    def _write(self, data):
        if False:
            return 10
        self._buf.write(data)

    def write_byte(self, byte):
        if False:
            print('Hello World!')
        self._buf.write(byte)

    def write_boolean(self, b):
        if False:
            print('Hello World!')
        enc = struct.pack('B', 1 if b else 0)
        self.write_byte(enc)

    def write_unsigned_short(self, s):
        if False:
            for i in range(10):
                print('nop')
        enc = struct.pack('>H', s)
        self._write(enc)

    def write_int(self, i):
        if False:
            print('Hello World!')
        enc = struct.pack('>i', i)
        self._write(enc)

    def write_long(self, l):
        if False:
            return 10
        enc = struct.pack('>Q', l)
        self._write(enc)

    def write_utf(self, s):
        if False:
            while True:
                i = 10
        utf = s.encode('utf8')
        byte_len = len(utf)
        if byte_len > 65535:
            raise OverflowError('UTF string may not exceed 65535 bytes!')
        self.write_unsigned_short(byte_len)
        self._write(utf)

    def finish(self):
        if False:
            return 10
        with BytesIO() as track_buf:
            byte_len = self._buf.getbuffer().nbytes
            flags = byte_len | 1 << 30
            enc_flags = struct.pack('>i', flags)
            track_buf.write(enc_flags)
            self._buf.seek(0)
            track_buf.write(self._buf.read())
            self._buf.close()
            track_buf.seek(0)
            return track_buf.read()