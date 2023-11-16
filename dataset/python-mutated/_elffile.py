"""
ELF file parser.

This provides a class ``ELFFile`` that parses an ELF executable in a similar
interface to ``ZipFile``. Only the read interface is implemented.

Based on: https://gist.github.com/lyssdod/f51579ae8d93c8657a5564aefc2ffbca
ELF header: https://refspecs.linuxfoundation.org/elf/gabi4+/ch4.eheader.html
"""
import enum
import os
import struct
from typing import IO, Optional, Tuple

class ELFInvalid(ValueError):
    pass

class EIClass(enum.IntEnum):
    C32 = 1
    C64 = 2

class EIData(enum.IntEnum):
    Lsb = 1
    Msb = 2

class EMachine(enum.IntEnum):
    I386 = 3
    S390 = 22
    Arm = 40
    X8664 = 62
    AArc64 = 183

class ELFFile:
    """
    Representation of an ELF executable.
    """

    def __init__(self, f: IO[bytes]) -> None:
        if False:
            i = 10
            return i + 15
        self._f = f
        try:
            ident = self._read('16B')
        except struct.error:
            raise ELFInvalid('unable to parse identification')
        magic = bytes(ident[:4])
        if magic != b'\x7fELF':
            raise ELFInvalid(f'invalid magic: {magic!r}')
        self.capacity = ident[4]
        self.encoding = ident[5]
        try:
            (e_fmt, self._p_fmt, self._p_idx) = {(1, 1): ('<HHIIIIIHHH', '<IIIIIIII', (0, 1, 4)), (1, 2): ('>HHIIIIIHHH', '>IIIIIIII', (0, 1, 4)), (2, 1): ('<HHIQQQIHHH', '<IIQQQQQQ', (0, 2, 5)), (2, 2): ('>HHIQQQIHHH', '>IIQQQQQQ', (0, 2, 5))}[self.capacity, self.encoding]
        except KeyError:
            raise ELFInvalid(f'unrecognized capacity ({self.capacity}) or encoding ({self.encoding})')
        try:
            (_, self.machine, _, _, self._e_phoff, _, self.flags, _, self._e_phentsize, self._e_phnum) = self._read(e_fmt)
        except struct.error as e:
            raise ELFInvalid('unable to parse machine and section information') from e

    def _read(self, fmt: str) -> Tuple[int, ...]:
        if False:
            return 10
        return struct.unpack(fmt, self._f.read(struct.calcsize(fmt)))

    @property
    def interpreter(self) -> Optional[str]:
        if False:
            return 10
        '\n        The path recorded in the ``PT_INTERP`` section header.\n        '
        for index in range(self._e_phnum):
            self._f.seek(self._e_phoff + self._e_phentsize * index)
            try:
                data = self._read(self._p_fmt)
            except struct.error:
                continue
            if data[self._p_idx[0]] != 3:
                continue
            self._f.seek(data[self._p_idx[1]])
            return os.fsdecode(self._f.read(data[self._p_idx[2]])).strip('\x00')
        return None