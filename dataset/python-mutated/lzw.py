import logging
from io import BytesIO
from typing import BinaryIO, Iterator, List, Optional, cast
logger = logging.getLogger(__name__)

class CorruptDataError(Exception):
    pass

class LZWDecoder:

    def __init__(self, fp: BinaryIO) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.fp = fp
        self.buff = 0
        self.bpos = 8
        self.nbits = 9
        self.table: List[Optional[bytes]] = []
        self.prevbuf: Optional[bytes] = None

    def readbits(self, bits: int) -> int:
        if False:
            i = 10
            return i + 15
        v = 0
        while 1:
            r = 8 - self.bpos
            if bits <= r:
                v = v << bits | self.buff >> r - bits & (1 << bits) - 1
                self.bpos += bits
                break
            else:
                v = v << r | self.buff & (1 << r) - 1
                bits -= r
                x = self.fp.read(1)
                if not x:
                    raise EOFError
                self.buff = ord(x)
                self.bpos = 0
        return v

    def feed(self, code: int) -> bytes:
        if False:
            while True:
                i = 10
        x = b''
        if code == 256:
            self.table = [bytes((c,)) for c in range(256)]
            self.table.append(None)
            self.table.append(None)
            self.prevbuf = b''
            self.nbits = 9
        elif code == 257:
            pass
        elif not self.prevbuf:
            x = self.prevbuf = cast(bytes, self.table[code])
        else:
            if code < len(self.table):
                x = cast(bytes, self.table[code])
                self.table.append(self.prevbuf + x[:1])
            elif code == len(self.table):
                self.table.append(self.prevbuf + self.prevbuf[:1])
                x = cast(bytes, self.table[code])
            else:
                raise CorruptDataError
            table_length = len(self.table)
            if table_length == 511:
                self.nbits = 10
            elif table_length == 1023:
                self.nbits = 11
            elif table_length == 2047:
                self.nbits = 12
            self.prevbuf = x
        return x

    def run(self) -> Iterator[bytes]:
        if False:
            print('Hello World!')
        while 1:
            try:
                code = self.readbits(self.nbits)
            except EOFError:
                break
            try:
                x = self.feed(code)
            except CorruptDataError:
                break
            yield x
            logger.debug('nbits=%d, code=%d, output=%r, table=%r', self.nbits, code, x, self.table[258:])

def lzwdecode(data: bytes) -> bytes:
    if False:
        print('Hello World!')
    fp = BytesIO(data)
    s = LZWDecoder(fp).run()
    return b''.join(s)